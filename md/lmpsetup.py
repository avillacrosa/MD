import numpy as np
import stat
import os
import mdtraj as md
import multiprocessing as mp
import pathlib
from string import Template
from md import hpssetup
from subprocess import run, PIPE


class LMPSetup(hpssetup.HPSSetup):
    """
    LMPSetup serves as a way to create easily LAMMPS input files (.lmp) as well as LAMMPS topology files (.data)
    """
    # TODO : Allow to pass parameters as kwargs
    def __init__(self, md_dir, protein, chains=1, model='HPS', **kwargs):
        super().__init__(md_dir, protein, model, chains, **kwargs)

    def del_missing_aas(self):
        """
        Do not include aminoacids neither in the pair_coeff entry in lmp.lmp and also not in data.data. It might be
        useful when debugging.
        :return: Nothing
        """
        missing, pos = [], []
        for key in self.residue_dict:
            if key not in self.sequence:
                pos.append(self.residue_dict[key]["id"])
                missing.append(key)

        for m in missing:
            del self.residue_dict[m]

        pos.reverse()

        for p in pos:
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] > p:
                    self.residue_dict[key]["id"] += -1
        self.key_ordering = list(self.residue_dict.keys())

    def lammps_ordering(self):
        """
        Aminoacids are ordered by alphabetical index by default. This method aminoacid indexing to a LAMMPS ordering,
        where each aminoacid is indexed by its order of appearence from the start of the chain
        :return: Nothing
        """
        idd = 1
        done_aas = []
        for aa in self.sequence:
            if aa not in done_aas:
                self.residue_dict[aa]["id"] = idd
                done_aas.append(aa)
                idd += 1
        ordered_keys = sorted(self.residue_dict, key=lambda x: (self.residue_dict[x]['id']))
        self.key_ordering = ordered_keys

    def make_equilibration_traj(self, t=100000):
        """
        Generate an equilibration trajectory
        :param t: How many steps do we want to run
        :return: Nothing
        """
        print(f"-> Equilibrating structure with a short LAMMPS run...")
        meta_maker = LMPSetup(oliba_wd=os.path.join(self.this, 'temp'), protein=self.protein, temper=False, equil_start=False)
        meta_maker.t = t
        meta_maker.save = int(t/10)
        meta_maker.temperatures = [300]
        meta_maker.write_hps_files(qsub=False, silent=True)
        meta_maker.run('lmp0.lmp', n_cores=8)
        traj = md.load(os.path.join(self.this, 'temp/dcd_traj_0.dcd'), top=os.path.join(self.this, 'temp/topo.pdb'))
        self.xyz = traj[-1].xyz * 10
        mx = np.abs(self.xyz).max()
        self.box_size["x"] = int(mx * 3)
        self.box_size["y"] = int(mx * 3)
        self.box_size["z"] = int(mx * 3)

        print(f"-> Saving equilibration pdb at {os.path.join(self.this, f'md/data/equil/{self.protein}.pdb')}")
        traj[-1].save_pdb(os.path.join(self.this, f'md/data/equil/{self.protein}.pdb'))

    def run(self, file, n_cores=1):
        """
        Run a LAMMPS job
        :param file: string, path to the .lmp file to run LAMMPS
        :param n_cores: int, number of cores we wish to use
        :return: stdout of the LAMMPS run
        """
        f_name = os.path.join(self.md_dir, file)
        if n_cores > mp.cpu_count():
            raise SystemExit(f'Desired number of cores exceed available cores on this machine ({mp.cpu_count()})')
        if n_cores > 1:
            command = f'mpirun -n {n_cores} {self.lmp} -in {f_name}'
        elif n_cores == 1:
            command = f'{self.lmp} -in {f_name}'
        else:
            raise SystemExit('Invalid core number')
        old_wd = os.getcwd()
        os.chdir(self.md_dir)
        out = run(command.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        os.chdir(old_wd)
        # for proc in psutil.process_iter():
        #     if proc.name() == 'lmp':
        #         proc.kill()
        return out

    def _lammps_pair_potential(self, temp_K):
        lines = ['pair_coeff          *       *       0.000000   0.000    0.000000   0.000   0.000\n']
        lambdas, epsilons = self._get_interactions(temp_K=temp_K)
        sigmas = self._get_sigmas()
        for i in range(len(self.residue_dict)):
            for j in range(i, len(self.residue_dict)):
                rd = self.residue_dict
                res_i = list(rd.keys())[i]
                res_j = list(rd.keys())[j]
                lambda_ij = lambdas[res_i][res_j]
                sigma_ij = sigmas[res_i][res_j]
                eps_ij = epsilons[res_i][res_j]
                if rd[res_i]["q"] != 0 and rd[res_j]["q"] != 0:
                    cutoff = 35.00
                else:
                    cutoff = 0.0
                line = 'pair_coeff         {:2d}      {:2d}      {: .6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                    i + 1, j + 1, eps_ij, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                lines.append(line)
        return lines

    def write_hps_files(self, qsub=True, slurm=False, pdb=True, silent=False):
        pathlib.Path(self.md_dir).mkdir(parents=True, exist_ok=True)
        self._get_topo()

        if qsub:
            qr_path = os.path.join(self.md_dir, f'run_qsub.sh')
            qsub_run = open(qr_path, 'w+')
            st = os.stat(qr_path)
            os.chmod(qr_path, st.st_mode | stat.S_IEXEC)
        if slurm:
            sr_path = os.path.join(self.md_dir, f'run_slm.sh')
            slm_run = open(sr_path, 'w+')
            st = os.stat(sr_path)
            os.chmod(sr_path, st.st_mode | stat.S_IEXEC)

        for temp_K in self.temperatures:
            self._write_lmp(temp_K=temp_K)
            self._write_data()
            if qsub:
                self._write_qsub(temp_K=temp_K)
                qsub_run.write(f'qsub {self.job_name}_{temp_K:.0f}.qsub \n')
            if slurm:
                self._write_slurm(temp_K=temp_K)
                slm_run.write(f'sbatch {self.job_name}_{temp_K:.0f}.slm \n')
            if pdb:
                self._write_pdb()
        if not silent:
            self._success_message()

        if qsub:
            qsub_run.close()
        if slurm:
            slm_run.close()

    def _write_data(self):
        subst_dict = self._generate_data_input()
        topo_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_data.data'))
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(subst_dict)

        with open(os.path.join(self.md_dir, 'data.data'), 'tw') as fileout:
            fileout.write(topo_subst)

    def _write_lmp(self, temp_K, rerun=False):
        pair_coeffs = self._lammps_pair_potential(temp_K)
        subst_dict = self._generate_lmp_input(temp_K, pair_coeffs)

        # TODO : CURRENTLY DEPRECATED. You can contact me if you want it implemented again
        # if self.temper:
        #     lmp_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/replica/input_template.lmp'))
        #     lmp_template = Template(lmp_temp_file.read())
        #     lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)
        # else:

        preface_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_preface.lmp'))
        preface_template = Template(preface_file.read())
        preface_subst = preface_template.safe_substitute(subst_dict)
        if str(self.slab).lower() == 'npt':
            lmp_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_slab_npt.lmp'))
        elif self.slab:
            lmp_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_slab.lmp'))
        elif rerun:
            lmp_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_rerun.lmp'))
        else:
            lmp_temp_file = open(os.path.join(self.this, 'md/data/templates/lmp/hps_general.lmp'))

        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(subst_dict)
        if self.fix_region is not None:
            helper = []
            for c in range(self.chains):
                d = np.array(self.fix_region)
                d = d + len(self.sequence) * c
                helper.append(f'{d[0]:.0f}:{d[1]:.0f}')
            helper = ' '.join(helper)
            subst_dict["fix_group"] = helper
            fixed = open(os.path.join(self.this, 'md/data/templates/lmp/hps_fix_rigid.lmp'))
            fixed_template = Template(fixed.read())
            fixed_subst = fixed_template.safe_substitute(subst_dict)
        else:
            fixed_subst = ""
        lmp_subst = preface_subst + fixed_subst + lmp_subst
        with open(os.path.join(self.md_dir, f'lmp{temp_K:.0f}.lmp'), 'tw') as fileout:
            fileout.write(lmp_subst)

    def _write_qsub(self, temp_K):
        subst_dict = self._generate_qsub(perdiu_dir=self.md_dir.replace('/perdiux', ''), temp_K=temp_K)
        qsub_temp_file = open(os.path.join(self.this, 'md/data/templates/qsub_template.qsub'))
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(subst_dict)
        with open(os.path.join(self.md_dir, f'{self.job_name}_{temp_K:.0f}.qsub'), 'tw') as fileout:
            fileout.write(qsub_subst)

    def _write_slurm(self, temp_K):
        subst_dict = self._generate_slurm(temp_K=temp_K)
        slurm_temp_file = open(os.path.join(self.this, 'md/data/templates/slurm_template.slm'))
        slurm_template = Template(slurm_temp_file.read())
        slurm_subst = slurm_template.safe_substitute(subst_dict)
        with open(os.path.join(self.md_dir, f'{self.job_name}_{temp_K:.0f}.slm'), 'tw') as fileout:
            fileout.write(slurm_subst)

    def _generate_lmp_input(self, temp_K, pair_coeffs):
        """
        Generate a python dict containing all the chosen parameters. The python dict is necessary for the python
        Template substituion, otherwise it would be useless
        :param temp_K: float, temperature in K we wish to write. If coming from a REX, T is None
        :return: Nothing
        """
        lmp_dict = {}
        if self.model.lower() == 'ghps' or self.model.lower()=='ghps-t':
            lmp_dict["pair_potential"] = 'lj/cut/coul/debye'
            if self.debye:
                lmp_dict["pair_parameters"] = f"{self.debye} 0.0"
            else:
                lmp_dict["pair_parameters"] = f"{round(1 / self.debye_length(temp_K) * 10 ** -10, 3)} 0.0"
        elif self.model.lower() == 'kh':
            lmp_dict["pair_potential"] = 'kh/cut/coul/debye'
            lmp_dict["pair_parameters"] = f"{self.debye} 0.0 35.0"
        else:
            lmp_dict["pair_potential"] = 'ljlambda'
            if self.debye:
                lmp_dict["pair_parameters"] = f"{self.debye} 0.0 35.0"
            else:
                lmp_dict["pair_parameters"] = f"{round(1 / self.debye_length(temp_K, ) * 10 ** -10, 3)} 0.0 35.0"


        dcd_dump = f"dcd_traj_{temp_K:.0f}.dcd"
        lammps_dump = f"atom_traj_{temp_K:.0f}.lammpstrj"
        log_file = f"log_{temp_K:.0f}.lammps"

        lmp_dict["t"] = self.t
        lmp_dict["dt"] = self.dt
        lmp_dict["pair_coeff"] = ''.join(pair_coeffs)
        lmp_dict["v_seed"] = self.v_seed
        lmp_dict["langevin_seed"] = self.langevin_seed
        if self.temper:
            lmp_dict["temperatures"] = ' '.join(map(str, self.temperatures))
        else:
            lmp_dict["temp"] = temp_K
        # TODO : Remember to add funcionality when we want constant EPS
        # self.lmp_file_dict["water_perm"] = self.water_perm
        if self.use_temp_eps:
            lmp_dict["water_perm"] = self._eps(temp_K)
        else:
            lmp_dict["water_perm"] = self.water_perm
        lmp_dict["swap_every"] = self.swap_every
        lmp_dict["save"] = self.save
        lmp_dict["rerun_skip"] = self.rerun_skip
        lmp_dict["rerun_start"] = self.rerun_start
        lmp_dict["rerun_stop"] = self.rerun_stop
        if int(self.t / 10000) != 0:
            lmp_dict["restart"] = int(self.t / 10000)
        else:
            lmp_dict["restart"] = 500
        # TODO this sucks but it is what it is, better option upstairs..
        ntemps = len(self.temperatures)
        lmp_dict["replicas"] = ' '.join(map(str, np.linspace(0, ntemps - 1, ntemps, dtype='int')))
        lmp_dict["rerun_dump"] = self.rerun_dump
        lmp_dict["langevin_damp"] = self.langevin_damp
        lmp_dict["deformation_ts"] = self.deformation_ts

        lmp_dict["final_slab_x"] = round(self.slab_dimensions["x"]/2, 2)
        lmp_dict["final_slab_y"] = round(self.slab_dimensions["y"]/2, 2)
        lmp_dict["final_slab_z"] = round(self.slab_dimensions["z"]/2, 2)

        lmp_dict["lammps_dump"] = lammps_dump
        lmp_dict["hps_scale"] = self.hps_scale
        lmp_dict["dcd_dump"] = dcd_dump
        lmp_dict["log_file"] = log_file

        lmp_dict["slab_t"] = self.slab_t
        return lmp_dict

    def _generate_data_input(self):
        """
        Generate the topology dict containing all chosen parameters. The dict is necessary for Python Template
        substitution
        :return:
        """

        masses = []
        data_dict = {}
        for i in range(1, len(self.residue_dict) + 1):
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] == i:
                    masses.append(f'          {i:2d}   {self.residue_dict[key]["mass"]}        #{key} \n')

        atoms, bonds = [], []
        k = 1
        for chain in range(1, self.chains + 1):
            for aa in self.sequence:
                xyz = self.xyz[0, k - 1, :]
                atoms.append(f'     {k :3d}          {chain}    '
                             f'     {self.residue_dict[aa]["id"]:2d}   '
                             f'   {self.residue_dict[aa]["q"]*self.charge_scale: .2f}'
                             f'    {xyz[0]: <6.3f}'
                             f'    {xyz[1]: .3f}'
                             f'    {xyz[2]: .3f}                 #{aa} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'     {k:3d}       1     {k:3d}     {k + 1:3d}\n')
                k += 1
        data_dict["natoms"] = self.chains * len(self.sequence)
        data_dict["nbonds"] = self.chains * (len(self.sequence) - 1)
        data_dict["atom_types"] = len(self.residue_dict)
        data_dict["masses"] = ''.join(masses)
        data_dict["atoms"] = ''.join(atoms)
        data_dict["bonds"] = ''.join(bonds)
        data_dict["box_size_x"] = int(self.box_size["x"]/2)
        data_dict["box_size_y"] = int(self.box_size["y"]/2)
        data_dict["box_size_z"] = int(self.box_size["z"]/2)
        return data_dict

    def _generate_qsub(self, perdiu_dir, temp_K):
        """
        Generate a qsub file to run at perdiux's
        :return:
        """
        qsub_dict = {}
        qsub_dict["work_dir"] = perdiu_dir
        # TODO : ADD STRING INSTEAD
        if self.temper:
            if self.processors == 1:
                qsub_dict[
                    "command"] = f"/home/adria/local/lammps3/bin/lmp -in lmp{temp_K:.0f}.lmp"
            else:
                qsub_dict[
                    "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps3/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            qsub_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps3/bin/lmp -in lmp{temp_K:.0f}.lmp -log log_{temp_K:.0f}.lammps"
        qsub_dict["np"] = self.processors
        qsub_dict["host"] = self.host
        qsub_dict["jobname"] = self.job_name
        return qsub_dict

    def _generate_slurm(self, temp_K):
        """
        Generate a slm file to run at CSUC
        :return:
        """
        slurm_dict = {}
        if self.temper:
            slurm_dict["command"] = f"srun `which lmp` -in lmp.lmp -partition {self.processors}x1"
        else:
            slurm_dict["command"] = f"srun `which lmp` -in lmp{temp_K:.0f}.lmp -log log_{temp_K:.0f}.lammps"
        slurm_dict["np"] = self.processors
        slurm_dict["jobname"] = self.job_name
        slurm_dict["in_files"] = f"data.data lmp{temp_K:.0f}.lmp"
        slurm_dict["temp"] = f"{temp_K:.0f}"
        return slurm_dict