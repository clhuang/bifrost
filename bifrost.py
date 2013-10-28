#
# Set of programs to read and interact with output from Bifrost
#

import numpy as N
import os
import re

class OSC_data:
    def __init__(self, snap, template='qsmag-by00_t%03i', meshfile=None, fdir='.',
                 verbose=True,dtype='f4', big_endian=False):
        ''' Main object for extracting Bifrost datacubes. '''

        self.snap     = snap
        self.fdir     = fdir
        self.template = fdir+'/'+template % (snap)
        self.verbose  = verbose

        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype

        # read idl file
        self.read_params()

        # read mesh file
        if meshfile is None:
            meshfile = fdir + '/' + self.params['meshfile'].strip()

        if not os.path.isfile(meshfile):
            raise IOError('Mesh file %s does not exist, aborting.' % meshfile)

        self.read_mesh(meshfile)

        # variables: lists and initialisation
        self.auxvars  = self.params['aux'].split()
        self.snapvars = ['r', 'px', 'py', 'pz', 'e', 'bx', 'by', 'bz']
        self.hionvars = ['hionne','hiontg','n1','n2','n3','n4','n5','n6','fion','nh2']
        self.compvars = ['ux', 'uy', 'uz', 'ee', 's'] # composite variables

        self.init_vars()


        return
    #------------------------------------------------------------------------

    def read_params(self):
        ''' Reads parameter file (.idl) '''

        filename = self.template + '.idl'
        self.params = read_idl_ascii(filename)

        # assign some parameters to root object
        try:
            self.nx = self.params['mx']
        except KeyError:
            raise KeyError('read_params: could not find nx in idl file!')

        try:
            self.ny = self.params['my']
        except KeyError:
            raise KeyError('read_params: could not find ny in idl file!')

        try:
            self.nz = self.params['mz']
        except KeyError:
            raise KeyError('read_params: could not find nz in idl file!')


        # check if units are there, if not use defaults and print warning
        unit_def = {'u_l': 1.e8, 'u_t': 1.e2, 'u_r': 1.e-7, 'u_b': 1.121e3, 'u_ee': 1.e12}

        for unit in unit_def:
            if not self.params.has_key(unit):
                print("(WWW) read_params: %s not found, using default of %.3e" %
                        (unit, unit_def[unit]))
                self.params[unit] = unit_def[unit]

        return

    #------------------------------------------------------------------------

    def read_mesh(self, meshfile):
        ''' Reads mesh.dat file '''

        # perhaps one day we'll be able to use N.genfromtxt, but for now
        # doing manually

        f = open(meshfile,'r')

        mx = int(f.readline().strip('\n').strip())
        assert mx == self.nx

        self.x       = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.xdn     = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dxidxup = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dxidxdn = N.array([float(v) for v in f.readline().strip('\n').split()])

        my = int(f.readline().strip('\n').strip())
        assert my == self.ny

        self.y       = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.ydn     = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dyidyup = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dyidydn = N.array([float(v) for v in f.readline().strip('\n').split()])

        mz = int(f.readline().strip('\n').strip())
        assert mz == self.nz

        self.z       = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.zdn     = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dyidzup = N.array([float(v) for v in f.readline().strip('\n').split()])
        self.dyidzdn = N.array([float(v) for v in f.readline().strip('\n').split()])

        f.close()

        return

    #------------------------------------------------------------------------
    def getvar(self, var, slice=None, order='F'):
        ''' Reads a given variable from the relevant files. '''

        import os

        if var == 'x':
            return self.x
        elif var == 'y':
            return self.y
        elif var == 'z':
            return self.z

        # find in which file the variable is
        if var in self.compvars:
            # if variable is composite, use getcompvar
            return self.getcompvar(var,slice)
        elif var in self.snapvars:
            fsuffix = '.snap'
            idx = self.snapvars.index(var)
            filename = self.template + fsuffix
        elif var in self.auxvars:
            fsuffix = '.aux'
            idx = self.auxvars.index(var)
            filename = self.template + fsuffix
        elif var in self.hionvars:
            idx = self.hionvars.index(var)
            isnap = self.params['isnap']
            if isnap <= -1:
                fsuffix = '.hion.snap.scr'
                filename = self.template + fsuffix
            elif isnap == 0:
                fsuffix = '.hion.snap'
                filename = self.template + fsuffix
            elif isnap > 0:
                fsuffix = '.hion_%03i.snap' % self.params['isnap']
                filename = '%s.hion%s.snap' % (self.template.split(str(isnap))[0], isnap)

            if not os.path.isfile(filename):
                filename = self.template + '.snap'
        elif re.match('ion[0-9]+', var): #is OOEVar
            idx = int(var[3:])
            fsuffix = '.ooe.snap'
            filename = self.template + fsuffix
            if os.stat(filename).st_size < self.nx*self.ny*self.nz*(idx+1)*4:
                raise ValueError('OOEVar level out of range.')
        else:
            raise ValueError('getvar: variable %s not available. Available vars:'
                  % (var) + '\n' + repr(self.auxvars + self.snapvars + self.hionvars + self.compvars))

        # Now memmap the variable
        if not os.path.isfile(filename):
            raise IOError('getvar: variable %s should be in %s file, not found!' %
                            (var, filename))

        # size of the data type
        if self.dtype[1:] == 'f4':
            dsize = 4
        else:
            raise ValueError('getvar: datatype %s not supported' % self.dtype)

        offset = self.nx*self.ny*self.nz*idx*dsize

        return N.memmap(filename, dtype=self.dtype,order=order, offset=offset,
                        mode='r', shape=(self.nx,self.ny,self.nz))

    #-----------------------------------------------------------------------

    def getcompvar(self,var,slice=None):
        ''' Gets composite variables. '''

        import cstagger

        # if rho is not loaded, do it (essential for composite variables)
        # rc is the same as r, but in C order (so that cstagger works)
        if not hasattr(self,'rc'):
            self.rc = self.variables['rc'] = self.getvar('r',order='C')
            # initialise cstagger
            rdt = self.r.dtype
            cstagger.init_stagger(self.nz, self.z.astype(rdt), self.zdn.astype(rdt))

        if   var == 'ux':  # x velocity
            if not hasattr(self,'px'):  self.px=self.variables['px']=self.getvar('px')
            return self.px/cstagger.xdn(self.rc)

        elif var == 'uy':  # y velocity
            if not hasattr(self,'py'): self.py=self.variables['py']=self.getvar('py')
            return self.py/cstagger.ydn(self.rc)

        elif var == 'uz':  # z velocity
            if not hasattr(self,'pz'): self.pz=self.variables['pz']=self.getvar('pz')
            return self.pz/cstagger.zdn(self.rc)

        elif var == 'ee':   # internal energy?
            if not hasattr(self,'e'): self.e=self.variables['e']=self.getvar('e')
            return self.e/self.r

        elif var == 's':   # entropy?
            if not hasattr(self,'p'): self.p=self.variables['p']=self.getvar('p')
            return N.log(self.p) - 1.667*N.log(self.r)
        else:
            raise ValueError('getcompvar: composite var %s not found. Available:\n %s'
                             % (var, repr(self.compvars)))


        return

    #-----------------------------------------------------------------------

    def getooevar(self,level,slice=None):
        ''' Gets ion data. level is the ionization level number'''
        return self.getvar('ion' + str(level))

    #-----------------------------------------------------------------------

    def init_vars(self):
        ''' Memmaps aux and snap variables, and maps them to methods. '''

        self.variables = {}

        # snap variables
        for var in self.snapvars + self.auxvars:
            self.variables[var] = self.getvar(var)
            setattr(self,var,self.variables[var])

        return

    #-----------------------------------------------------------------------

    def write_rh15d(self, outfile, sx=None, sy=None, sz=None, desc=None,
                    append=True):
        ''' Writes RH 1.5D NetCDF snapshot '''

        from tt.lines import rh15d

        # unit conversion to SI
        ul = self.params['u_l'] / 1.e2 # to metres
        ur = self.params['u_r']        # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']        # to seconds
        uv = ul/ut
        ub = self.params['u_b'] * 1e-4 # to Tesla
        ue = self.params['u_ee']       # to erg/g

        # slicing and unit conversion
        if sx is None: sx = [0, self.nx, 1]
        if sy is None: sy = [0, self.ny, 1]
        if sz is None: sz = [0, self.nz, 1]

        hion = False

        if self.params.has_key('do_hion'):
            if self.params['do_hion'] > 0:
                hion = True

        print('Slicing and unit conversion...')
        temp = self.tg[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho  =  self.r[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho  = rho*ur
        Bx   = self.bx[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        By   = self.by[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        Bz   = self.bz[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        Bx   = Bx*ub; By = By*ub; Bz = -Bz*ub # CHANGED BZ TO MINUS RECENTLY!!!!!
        vz=self.getcompvar('uz')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        vz  *= -uv


        x    = self.x[sx[0]:sx[1]:sx[2]]*ul
        y    = self.y[sy[0]:sy[1]:sy[2]]*ul
        z    = self.z[sz[0]:sz[1]:sz[2]]*(-ul)

        # convert from rho to H atoms, ideally from subs.dat. Otherwise use default.
        if hion:
            print('Getting hion data...')
            ne = self.getvar('hionne')
            # slice and convert from cm^-3 to m^-3
            ne  = ne[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
            ne  = ne * 1.e6

            # read hydrogen populations (they are saved in cm^-3)
            nh = N.empty((6,) + temp.shape, dtype='Float32')

            for k in range(6):
                nv = self.getvar('n%i' % (k+1))
                nh[k] = nv[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
            nh = nh * 1.e6

        else:
            ee=self.getcompvar('ee')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
            ee = ee * ue

            if os.path.isfile('%s/subs.dat' % self.fdir):
                grph = subs2grph('%s/subs.dat' % self.fdir)
            else:
                grph = 2.380491e-24

            nh = rho/grph * 1.e6       # from rho to nH in m^-3

            # interpolate ne from the EOS table
            print('ne interpolation...')
            eostab = Rhoeetab(fdir=self.fdir)
            ne = eostab.tab_interp(rho, ee, order=1) * 1.e6  # from cm^-3 to m^-3

            ## old method, using Mats's table
            # ne = ne_rt_table(rho, temp) * 1.e6  # from cm^-3 to m^-3

        # description
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.template, repr(sx), repr(sy), repr(sz))
            if hion:
                desc = 'hion '+desc

        # write to file
        print('Write to file...')
        rh15d.make_ncdf_atmos(outfile, temp, vz, ne, nh, z, x=x, y=y, append=append,
                              Bx=Bx, By=By, Bz=Bz, desc=desc, snap=self.snap)

        return

    #-------------------------------------------------------------------------------------

    def write_rh15d_simple(self, outfile, sx=None, sy=None, sz=None, desc=None,
                    append=True, writeB=True):
        ''' Writes RH 1.5D NetCDF snapshot with no compression in a memory-saving manner.'''

        import netCDF4 as nc

        # unit conversion to SI
        ul = self.params['u_l'] / 1.e2 # to metres
        ur = self.params['u_r']        # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']        # to seconds
        uv = ul/ut
        ub = self.params['u_b'] * 1e-4 # to Tesla
        ue = self.params['u_ee']       # to erg/g

        # slicing and unit conversion
        if sx is None: sx = [0, self.nx, 1]
        if sy is None: sy = [0, self.ny, 1]
        if sz is None: sz = [0, self.nz, 1]

        vars = {'temperature': [self.tg, 1.],
                'velocity_z': None,
                'electron_density': None,
                'hydrogen_populations': None,
                'B_x': [self.bx, ub],
                'B_y': [self.by, ub],
                'B_z': [self.bz, ub] }

        hion = False

        if self.params.has_key('do_hion'):
            if self.params['do_hion'] > 0:
                hion = True

        # find dimensions to write
        nx = len(range(sx[0],sx[1],sx[2]))
        ny = len(range(sy[0],sy[1],sy[2]))
        nz = len(range(sz[0],sz[1],sz[2]))

        if hion:
            nhydr = 6
        else:
            nhydr = 1

        # description
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.template, repr(sx), repr(sy), repr(sz))
            if hion:
                desc = 'hion '+desc

        # create netCDF file
        mode = ['w','a']
        if (append and not os.path.isfile(outfile)): append=False

        rootgrp = nc.Dataset(outfile, mode[append], format='NETCDF4')

        if not append:
            rootgrp.createDimension('nt', None) # create unlimited dimension
            rootgrp.createDimension('nx', nx)
            rootgrp.createDimension('ny', ny)
            rootgrp.createDimension('nz', nz)
            rootgrp.createDimension('nhydr', nhydr)

            rootgrp.createVariable('temperature', 'f4', ('nt','nx','ny','nz'),
                                            least_significant_digit=1)
            rootgrp.createVariable('velocity_z', 'f4', ('nt','nx','ny','nz'),
                                            least_significant_digit=1)
            rootgrp.createVariable('electron_density', 'f8', ('nt','nx','ny','nz'))
            rootgrp.createVariable('hydrogen_populations', 'f4',
                                            ('nt','nhydr','nx','ny','nz'))

            x_var  = rootgrp.createVariable('x', 'f4', ('nx',))
            y_var  = rootgrp.createVariable('y', 'f4', ('ny',))
            z_var  = rootgrp.createVariable('z', 'f4', ('nt','nz'))
            nt_var = rootgrp.createVariable('snapshot_number', 'i4', ('nt',))

            rootgrp.description = desc
            rootgrp.has_B = 0

            if writeB:
                rootgrp.createVariable('B_x', 'f4', ('nt','nx','ny','nz'),
                                       least_significant_digit=5)
                rootgrp.createVariable('B_y', 'f4', ('nt','nx','ny','nz'),
                                       least_significant_digit=5)
                rootgrp.createVariable('B_z', 'f4', ('nt','nx','ny','nz'),
                                       least_significant_digit=5)
                rootgrp.has_B = 1

            nt = [0, 1]
        else:
            x_var = rootgrp.variables['x']
            y_var = rootgrp.variables['y']
            z_var = rootgrp.variables['z']

            nti = len(rootgrp.dimensions['nt'])
            nt  = [nti, nti+nt]


        # write small arrays
        x_var[:] = self.x[sx[0]:sx[1]:sx[2]]*ul
        y_var[:] = self.y[sy[0]:sy[1]:sy[2]]*ul
        z_var[:] = self.z[N.newaxis, sz[0]:sz[1]:sz[2]]*(-ul)
        nt_var[nt[0]:nt[1]] = self.snap

        # write large arrays, one by one
        for v in vars:
            bufvar = rootgrp.variables[v]

            if v in ['B_x', 'B_y', 'B_z', 'temperature']:
                bufvar[nt[0]:nt[1]] = vars[v][0][N.newaxis,
                                                 sx[0]:sx[1]:sx[2],
                                                 sy[0]:sy[1]:sy[2],
                                                 sz[0]:sz[1]:sz[2]] * vars[v][1]
            elif v == 'velocity_z':
                bufvar[nt[0]:nt[1]] = self.getcompvar('uz')[N.newaxis,
                                                            sx[0]:sx[1]:sx[2],
                                                            sy[0]:sy[1]:sy[2],
                                                            sz[0]:sz[1]:sz[2]] * (-uv)
            elif v == 'electron_density':
                if hion:
                    # slice and convert from cm^-3 to m^-3
                    bufvar[nt[0]:nt[1]] = self.getvar('hionne')[N.newaxis,
                                                                sx[0]:sx[1]:sx[2],
                                                                sy[0]:sy[1]:sy[2],
                                                                sz[0]:sz[1]:sz[2]] * 1.e6
                else:
                    ee=self.getcompvar('ee')[sx[0]:sx[1]:sx[2],
                                             sy[0]:sy[1]:sy[2],
                                             sz[0]:sz[1]:sz[2]] * ue

                    rho =  self.r[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
                    rho = rho*ur

                    print('ne interpolation...')
                    eostab = Rhoeetab(fdir=self.fdir)
                    ne = eostab.tab_interp(rho, ee, order=1) * 1.e6  # from cm^-3 to m^-3

                    bufvar[nt[0]:nt[1]] = ne[N.newaxis, :]

            elif v == 'hydrogen_populations':
                if hion:
                    # read hydrogen populations (they are saved in cm^-3)
                    nh = N.empty((6,) + temp.shape, dtype='Float32')

                    for k in range(6):
                        nv = self.getvar('n%i' % (k+1))
                        nh[k] = nv[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
                    nh = nh * 1.e6

                    bufvar[nt[0]:nt[1]] = nh[N.newaxis,:]
                else:
                    if os.path.isfile('%s/subs.dat' % self.fdir):
                        grph = subs2grph('%s/subs.dat' % self.fdir)
                    else:
                        grph = 2.380491e-24

                    # from rho to nH in m^-3
                    bufvar[nt[0]:nt[1]] = self.r[N.newaxis,
                                                 sx[0]:sx[1]:sx[2],
                                                 sy[0]:sy[1]:sy[2],
                                                 sz[0]:sz[1]:sz[2]] * ur / grph * 1.e6
        rootgrp.close()

        return

#-----------------------------------------------------------------------------------------

class Rhoeetab:
    def __init__(self, tabfile=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True, radtab=False):

        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian

        self.eosload = False
        self.radload = False

        # read table file and calculate parameters
        if tabfile is None:
            tabfile = '%s/tabparam.in' % (fdir)
        self.param = self.read_tab_file(tabfile)

        # load table(s)
        self.load_eos_table()
        if radtab: self.load_rad_table()

        return

    #-------------------------------------------------------------------------------------

    def read_tab_file(self,tabfile):
        ''' Reads tabparam.in file, populates parameters. '''

        self.params = read_idl_ascii(tabfile)
        if self.verbose: print('*** Read parameters from '+tabfile)

        p = self.params

        # construct lnrho array
        self.lnrho = N.linspace(N.log(p['rhomin']), N.log(p['rhomax']), p['nrhobin'])
        self.dlnrho= self.lnrho[1] - self.lnrho[0]

        # construct ei array
        self.lnei = N.linspace(N.log(p['eimin']), N.log(p['eimax']), p['neibin'])
        self.dlnei= self.lnei[1] - self.lnei[0]


        return

    #-------------------------------------------------------------------------------------

    def load_eos_table(self, eostabfile=None):
        ''' Loads EOS table. '''

        if eostabfile is None:
            eostabfile = '%s/%s' % (self.fdir, self.params['eostablefile'])

        nei  = self.params['neibin']
        nrho = self.params['nrhobin']

        dtype = ('>' if self.big_endian else '<') + self.dtype

        table = N.memmap(eostabfile, mode='r', shape=(nei,nrho,4), dtype=dtype,
                         order='F')

        self.lnpg = table[:,:,0]
        self.tgt  = table[:,:,1]
        self.lnne = table[:,:,2]
        self.lnrk = table[:,:,3]

        self.eosload = True
        if self.verbose: print('*** Read EOS table from '+eostabfile)

        return

    #-------------------------------------------------------------------------------------

    def load_rad_table(self, radtabfile=None):
        ''' Loads rhoei_radtab table. '''

        if radtabfile is None:
            radtabfile = '%s/%s' % (self.fdir, self.params['rhoeiradtablefile'])

        nei  = self.params['neibin']
        nrho = self.params['nrhobin']
        nbins= self.params['nradbins']

        dtype = ('>' if self.big_endian else '<') + self.dtype

        table = N.memmap(radtabfile, mode='r', shape=(nei,nrho,nbins,3), dtype=dtype,
                         order='F')

        self.epstab = table[:,:,:,0]
        self.temtab = table[:,:,:,1]
        self.opatab = table[:,:,:,2]

        self.radload = True
        if self.verbose: print('*** Read rad table from '+radtabfile)

        return

    #-------------------------------------------------------------------------------------

    def get_table(self, out='ne', bine=None, order=1):
        import scipy.ndimage as ndimage

        qdict = {'ne':'lnne', 'tg':'tgt', 'pg':'lnpg', 'kr':'lnkr',
                 'eps':'epstab', 'opa':'opatab', 'temp':'temtab'  }

        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")

        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")

        quant = getattr(self, qdict[out])

        if out in ['opa eps temp'.split()]:
            if bin is None:
                print("(WWW) tab_interp: radiation bin not set, using first bin.")
                bin = 0
            quant = quant[:,:,bin]

        return quant


    def tab_interp(self, rho, ei, out='ne', bin=None, order=1):
        ''' Interpolates the EOS/rad table for the required quantity in out.

            IN:
                rho  : density [g/cm^3]
                ei   : internal energy [erg/g]
                bin  : (optional) radiation bin number for bin parameters
                order: interpolation order (1: linear, 3: cubic)

            OUT:
                depending on value of out:
                'nel'  : electron density [cm^-3]
                'tg'   : temperature [K]
                'pg'   : gas pressure [dyn/cm^2]
                'kr'   : Rosseland opacity [cm^2/g]
                'eps'  : scattering probability
                'opa'  : opacity
                'temt' : thermal emission
        '''

        import scipy.ndimage as ndimage

        qdict = {'ne':'lnne', 'tg':'tgt', 'pg':'lnpg', 'kr':'lnkr',
                 'eps':'epstab', 'opa':'opatab', 'temp':'temtab'  }

        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")

        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")

        quant = getattr(self, qdict[out])

        if out in ['opa eps temp'.split()]:
            if bin is None:
                print("(WWW) tab_interp: radiation bin not set, using first bin.")
                bin = 0
            quant = quant[:,:,bin]


        # warnings for values outside of table
        rhomin= N.min(rho) ; rhomax = N.max(rho)
        eimin = N.min(ei)  ; eimax  = N.max(ei)

        if rhomin < self.params['rhomin']:
            print('(WWW) tab_interp: density outside table bounds. ' +
                  'Table rho min=%.3e, requested rho min=%.3e' % (self.params['rhomin'],rhomin))
        if rhomax > self.params['rhomax']:
            print('(WWW) tab_interp: density outside table bounds. ' +
                  'Table rho max=%.1f, requested rho max=%.1f' % (self.params['rhomax'],rhomax))
        if eimin < self.params['eimin']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                 'Table Ei min=%.2f, requested Ei min=%.2f' % (self.params['eimin'],eimin))
        if eimax > self.params['eimax']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                 'Table Ei max=%.2f, requested Ei max=%.2f' % (self.params['eimax'],eimax))


        # translate to table coordinates
        x = (N.log(ei)  -  self.lnei[0]) / self.dlnei
        y = (N.log(rho) - self.lnrho[0]) / self.dlnrho

        # interpolate quantity
        result = ndimage.map_coordinates(quant, [x,y], order=order, mode='nearest')


        return (N.exp(result) if out != 'tg' else result)

#-----------------------------------------------------------------------------------------

###############################
####         TOOLS         ####
###############################

def read_idl_ascii(filename):
    ''' Reads IDL-formatted (command style) ascii file into dictionary '''

    li = 0
    params = {}

    # go through the file, add stuff to dictionary
    for line in file(filename):
        # ignore empty lines and comments
        line = line.strip()
        if len(line) < 1 :
            li += 1
            continue
        if line[0] == ';':
            li += 1
            continue

        line = line.split(';')[0].split('=')

        if (len(line) != 2):
            print('(WWW) read_params: line %i is invalid, continuing' % li)
            li += 1
            continue

        key   = line[0].strip().lower() # force lowercase because IDL is case-insensitive
        value = line[1].strip()

        # instead of the insecure 'exec', find out the datatypes
        if (value.find('"') >= 0):
            # string type
            value = value.strip('"')
        elif (value.find("'") >= 0):
            value = value.strip("'")
        elif (value.lower() in ['.false.', '.true.']):
            # bool type
            value = False if value.lower() == '.false.' else True
        elif ((value.upper().find('E') >= 0) or (value.find('.') >= 0 )):
            # float type
            value = float(value)
        elif (value.find('[') >= 0 and value.find(']') >= 0):
            # list type
            value = eval(value)
        else:
            # int type
            try:
                value = int(value)
            except:
                print('(WWW) read_idl_ascii: could not find datatype in line %i, skipping' % li)
                li += 1
                continue

        params[key] = value

        li += 1

    return params

#-----------------------------------------------------------------------------------------


def subs2grph(subsfile):
    ''' From a subs.dat file, extract abundances and atomic masses to calculate
    grph, grams per hydrogen. '''

    from scipy.constants import atomic_mass as amu

    f = open(subsfile,'r')
    nspecies = N.fromfile(f,count=1,sep=' ',dtype='i')[0]
    f.readline() # second line not important
    ab = N.fromfile(f, count=nspecies, sep=' ',dtype='f')
    am = N.fromfile(f, count=nspecies, sep=' ',dtype='f')
    f.close()

    # linear abundances
    ab = 10.**(ab-12.)

    # mass in grams
    am *= amu * 1.e3

    return N.sum(ab*am)

#-----------------------------------------------------------------------------------------

class Opatab:
    def __init__(self, tabname=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True,lambd=100.0):

        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.lambd = lambd
        self.radload = False
        self.teinit = 4.0
        self.dte = 0.1
        # read table file and calculate parameters
        if tabname is None:
            tabname = '%s/ionization.dat' % (fdir)

        self.tabname = tabname
        # load table(s)
        self.load_opa_table()

        return

#-----------------------------------------------------------------------------------------

    def hopac(self):
        ''' Calculates the photoionization cross sections given by
        from anzer & heinzel apj 622: 714-721, 2005, march 20
        these clowns have a couple of great big typos in their reported c's.... correct values to
        be found in rumph et al 1994 aj, 107: 2108, june 1994

        gaunt factors are set to 0.99 for h and 0.85 for heii, which should be good enough
        for the purposes of this code
        '''

        ghi = 0.99
        o0 = 7.91e-18 # cm^2

        ohi = 0
        if self.lambd <= 912:
            ohi = o0 * ghi * (self.lambd / 912.0)**3

        return ohi
#-----------------------------------------------------------------------------------------

    def heiopac(self):
        ''' Calculates the photoionization cross sections given by
        from anzer & heinzel apj 622: 714-721, 2005, march 20
        these clowns have a couple of great big typos in their reported c's.... correct values to
        be found in rumph et al 1994 aj, 107: 2108, june 1994

        gaunt factors are set to 0.99 for h and 0.85 for heii, which should be good enough
        for the purposes of this code
        '''

        c = [-2.953607e1, 7.083061e0, 8.678646e-1,
                -1.221932e0, 4.052997e-2, 1.317109e-1,
                -3.265795e-2, 2.500933e-3]

        ohei = 0
        if self.lambd <= 504:
            for i, cf in enumerate(c):
                ohei += cf * (N.log10(self.lambd))**i
            ohei = 10.0**ohei

        return ohei

#-----------------------------------------------------------------------------------------

    def heiiopac(self):
        ''' Calculates the photoionization cross sections given by
        from anzer & heinzel apj 622: 714-721, 2005, march 20
        these clowns have a couple of great big typos in their reported c's.... correct values to
        be found in rumph et al 1994 aj, 107: 2108, june 1994

        gaunt factors are set to 0.99 for h and 0.85 for heii, which should be good enough
        for the purposes of this code
        '''

        gheii = 0.85
        o0 = 7.91e-18 # cm^2

        oheii = 0
        if self.lambd <= 228:
            oheii = 16 * o0 * gheii * (self.lambd / 912.0)**3

        return oheii

#------------------------------------------------------------------------------------------

    def load_opa_table(self, tabname=None):
       ''' Loads ionizationstate table. '''

       if tabname is None:
           tabname = '%s/%s' % (self.fdir, 'ionization.dat')

       eostab = Rhoeetab(fdir=self.fdir)

       nei  = eostab.params['neibin']
       nrho = eostab.params['nrhobin']

       dtype = ('>' if self.big_endian else '<') + self.dtype

       table = N.memmap(tabname, mode='r', shape=(nei,nrho,3), dtype=dtype,
                        order='F')

       self.ionh = table[:,:,0]
       self.ionhe  = table[:,:,1]
       self.ionhei = table[:,:,2]

       self.opaload = True
       if self.verbose: print('*** Read EOS table from '+tabname)

       return

   #-------------------------------------------------------------------------------------

    def tg_tab_interp(self, order=1):
       ''' Interpolates the opa table to same format as tg table.
       '''

       import scipy.ndimage as ndimage

       self.load_opa1d_table()

       rhoeetab = Rhoeetab(fdir=self.fdir)
       tgTable = rhoeetab.get_table('tg')


       # translate to table coordinates
       x = (N.log10(tgTable)  -  self.teinit) / self.dte

       # interpolate quantity
       self.ionh = ndimage.map_coordinates(self.ionh1d, [x], order=order)#, mode='nearest')
       self.ionhe = ndimage.map_coordinates(self.ionhe1d, [x], order=order)#, mode='nearest')
       self.ionhei = ndimage.map_coordinates(self.ionhei1d, [x], order=order)#, mode='nearest')

       return

#-----------------------------------------------------------------------------------------

    def h_he_absorb(self,lambd=None):
   # ''' from anzer & heinzel apj 622: 714-721, 2005, march 2  '''

       rhe=0.1
       epsilon=1.e-20

       if lambd is not None:
           self.lambd = lambd

       self.tg_tab_interp()

       ion_h = self.ionh
       ion_he = self.ionhe

       ion_hei=self.ionhei

       ohi = self.hopac()
       ohei = self.heiopac()
       oheii = self.heiiopac()


       arr = (1 - ion_h) * ohi + rhe * ((1 - ion_he - ion_hei) * ohei + ion_he * oheii)
       arr[arr < 0] = 0
       '''
       Gets the opacities for a particular wavelength of light.
       If lambd is None, then looks at the current level for wavelength
       '''

       return arr

#----------------------------------------------------------------------------------------

    def load_opa1d_table(self, tabname=None):
       ''' Loads ionizationstate table. '''

       if tabname is None:
           tabname = '%s/%s' % (self.fdir, 'ionization1d.dat')

       dtype = ('>' if self.big_endian else '<') + self.dtype

       table = N.memmap(tabname, mode='r', shape=(41,3), dtype=dtype,
                        order='F')

       self.ionh1d = table[:,0]
       self.ionhe1d  = table[:,1]
       self.ionhei1d = table[:,2]

       self.opaload = True
       if self.verbose: print('*** Read OPA table from '+tabname)

       return

