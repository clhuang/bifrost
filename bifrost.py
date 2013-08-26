#
# Set of programs to read and interact with output from Bifrost
#

import numpy as N
import os

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
            meshfile = self.params['meshfile'].strip()

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
        elif isOOEVar(var):
            idx = int(var[3:])
            fsuffix = '.ooe.snap'
            filename = self.template + fsuffix
            if os.stat(filename).st_size < self.nx*self.ny*self.nz*(idx+1)*dsize:
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
        return getvar('lvl' + str(level))

    #-----------------------------------------------------------------------

    def isOOEVar(var):
        return re.match('ion[0-9]+', var)

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

def ne_rt_table(rho, temp, order=1, tabfile=None):
    ''' Calculates electron density by interpolating the rho/temp table.
        Based on Mats Carlsson's ne_rt_table.pro.

        IN: rho (in g/cm^3),
            temp (in K),

        OPTIONAL: order (interpolation order 1: linear, 3: cubic),
                  tabfile (path of table file)

        OUT: electron density (in g/cm^3)

        '''
    import os
    import scipy.interpolate as interp
    import scipy.ndimage as ndimage
    from scipy.io.idl import readsav


    print 'DEPRECATION WARNING: this method is deprecated in favour of the Rhoeetab class.'


    if tabfile is None:
        tabfile = 'ne_rt_table.idlsave'

    # use table in default location if not found
    if not os.path.isfile(tabfile) and \
        os.path.isfile(os.getenv('TIAGO_DATA')+'/misc/'+tabfile):
        tabfile = os.getenv('TIAGO_DATA')+'/misc/'+tabfile


    tt = readsav(tabfile, verbose=False)

    lgrho = N.log10(rho)

    # warnings for values outside of table
    tmin  = N.min(temp) ; tmax = N.max(temp)
    ttmin = N.min(5040./tt['theta_tab']) ; ttmax = N.max(5040./tt['theta_tab'])

    lrmin  = N.min(lgrho) ; lrmax = N.max(lgrho)
    tlrmin = N.min(tt['rho_tab']) ; tlrmax = N.max(tt['rho_tab'])

    if tmin < ttmin:
        print('(WWW) ne_rt_table: temperature outside table bounds. ' +
              'Table Tmin=%.1f, requested Tmin=%.1f' % (ttmin, tmin))
    if tmax > ttmax:
        print('(WWW) ne_rt_table: temperature outside table bounds. ' +
              'Table Tmax=%.1f, requested Tmax=%.1f' % (ttmax, tmax))
    if lrmin < tlrmin:
        print('(WWW) ne_rt_table: log density outside of table bounds. ' +
             'Table log(rho) min=%.2f, requested log(rho) min=%.2f' % (tlrmin, lrmin))
    if lrmax > tlrmax:
        print('(WWW) ne_rt_table: log density outside of table bounds. ' +
             'Table log(rho) max=%.2f, requested log(rho) max=%.2f' % (tlrmax, lrmax))

    ## Tiago: this is for the real thing, global fit 2D interpolation:
    ##        (commented because it is TREMENDOUSLY SLOW)
    #x = N.repeat(tt['rho_tab'],  tt['theta_tab'].shape[0])
    #y = N.tile(  tt['theta_tab'],  tt['rho_tab'].shape[0])
    ## 2D grid interpolation according to method (default: linear interpolation)
    #result = interp.griddata(N.transpose([x,y]), tt['ne_rt_table'].ravel(),
    #                         (lgrho, 5040./temp), method=method)
    #
    ## if some values outside of the table, use nearest neighbour
    #if N.any(N.isnan(result)):
    #    idx = N.isnan(result)
    #    near = interp.griddata(N.transpose([x,y]), tt['ne_rt_table'].ravel(),
    #                            (lgrho, 5040./temp), method='nearest')
    #    result[idx] = near[idx]

    ## Tiago: this is the approximate thing (bilinear/cubic interpolation) with ndimage
    y = (5040./temp - tt['theta_tab'][0])/(tt['theta_tab'][1]-tt['theta_tab'][0])
    x = (lgrho - tt['rho_tab'][0])/(tt['rho_tab'][1]-tt['rho_tab'][0])

    result=ndimage.map_coordinates(tt['ne_rt_table'], [x,y], order=order, mode='nearest')


    return 10**result*rho/tt['grph']
