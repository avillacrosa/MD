import numpy as np
import os
import inspect
import pathlib
import stat
from md import hpssetup
from string import Template


def HPS_potential(r, rmin, rmax, eps, lambd, sigma):
    V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    if r <= 2 ** (1 / 6) * sigma:
        V = V + (1 - lambd) * eps
    else:
        V = lambd * V
        F = lambd * F
    return (V, F)


def HPS_pi_potential(r, rmin, rmax, eps_hps, eps_pi, lambd, sigma):
    V = 4 * eps_hps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * eps_hps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    V_pi = eps_pi * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F_pi = eps_pi / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    if r <= 2 ** (1 / 6) * sigma:
        V = V + (1 - lambd) * eps_hps + V_pi
        F = F + F_pi
    else:
        V = lambd * V + V_pi
        F = lambd * F + F_pi
    return (V, F)


class HMDSetup(hpssetup.HPSSetup):
    def __init__(self, md_dir, protein, chains=1, model='HPS', compact=False, **kwargs):
        super().__init__(md_dir, protein, model, chains, **kwargs)
        self.compact = compact
        single_gpu = kwargs.get('single_gpu', False)
        if single_gpu:
            self.n_gpus = [0]*len(self.temperatures)
        else:
            self.n_gpus = np.arange(0,len(self.temperatures))

    def write_hps_files(self, qsub=True, slurm=False, silent=False):
        pathlib.Path(self.md_dir).mkdir(parents=True, exist_ok=True)
        self._get_topo()
        self._write_pdb()

        pr_path = os.path.join(self.md_dir, f'run_py.sh')
        py_run = open(pr_path, 'w+')
        st = os.stat(pr_path)
        os.chmod(pr_path, st.st_mode | stat.S_IEXEC)

        if slurm:
            sr_path = os.path.join(self.md_dir, f'run_slm.sh')
            slm_run = open(sr_path, 'w+')
            st = os.stat(sr_path)
            os.chmod(sr_path, st.st_mode | stat.S_IEXEC)

        for temp_K in self.temperatures:
            self._write_hmd(temp_K=temp_K)
            if qsub:
                self._write_qsub(temp_K=temp_K)
            if slurm:
                self._write_slurm(temp_K=temp_K)
                slm_run.write(f'sbatch {self.job_name}_{temp_K:.0f}.slm \n')
            py_run.write(f'python3 hmd_hps_{temp_K:.0f}.py > run_{temp_K:.0f}.log & \n')
        if not silent:
            self._success_message()

        if slurm:
            slm_run.close()

    def _write_hmd(self, temp_K):
        subst_dict = self._generate_hmd(temp_K=temp_K)
        lambdas_df, epsilons_df = self._get_interactions(temp_K=temp_K)
        sigmas = self._get_sigmas()
        if self.model.lower() == 'hps-cat':
            preface_file = open(os.path.join(self.this, 'md/data/templates/hmd/preface_pi.pyt'))
        elif self.compact:
            for aa in self.residue_dict:
                self.residue_dict[aa]["lambda"] = lambdas_df[aa][aa]
                self.residue_dict[aa]["sigma"] = sigmas[aa][aa]
            preface_file = open(os.path.join(self.this, 'md/data/templates/hmd/preface_compact.pyt'))
        else:
            preface_file = open(os.path.join(self.this, 'md/data/templates/hmd/preface.pyt'))
            self.residue_df.to_csv(os.path.join(self.md_dir, 'residues.txt'), sep=" ")
            lambdas_df.to_csv(os.path.join(self.md_dir, f'lambdas_{temp_K:.0f}.txt'), sep=" ")
            epsilons_df.to_csv(os.path.join(self.md_dir, 'epsilons.txt'), sep=" ")
            sigmas.to_csv(os.path.join(self.md_dir, 'sigmas.txt'), sep=" ")
        preface_template = Template(preface_file.read())
        preface_subst = preface_template.safe_substitute(subst_dict)

        if self.slab is True:
            hmd_temp_file = open(os.path.join(self.this, 'md/data/templates/hmd/slab.pyt'))
        elif self.slab_method.lower() == 'npt':
            hmd_temp_file = open(os.path.join(self.this, 'md/data/templates/hmd/slab_npt.pyt'))
        else:
            hmd_temp_file = open(os.path.join(self.this, 'md/data/templates/hmd/general.pyt'))
        hmd_template = Template(hmd_temp_file.read())
        hmd_subst = hmd_template.safe_substitute(subst_dict)
        hmd_file = preface_subst + hmd_subst
        with open(os.path.join(self.md_dir, f'hmd_hps_{temp_K:.0f}.py'), 'tw') as fileout:
            fileout.write(hmd_file)

    def _write_slurm(self, temp_K):
        subst_dict = self._generate_slurm(temp_K=temp_K)
        slurm_temp_file = open(os.path.join(self.this, 'md/data//templates/hmd_slurm_template.slm'))
        slurm_template = Template(slurm_temp_file.read())
        slurm_subst = slurm_template.safe_substitute(subst_dict)
        with open(os.path.join(self.md_dir, f'{self.job_name}_{temp_K:.0f}.slm'), 'tw') as fileout:
            fileout.write(slurm_subst)

    def _write_qsub(self, temp_K):
        # TODO
        pass

    def _generate_hmd(self, temp_K):
        hmd_dict = {}
        hmd_dict["temperature"] = temp_K
        hmd_dict["box_size"] = self.box_size["x"]/10.
        hmd_dict["particles"] = self.residue_dict
        hmd_dict["particle_types"] = list(self.residue_dict.keys())
        hmd_dict["chains"] = self.chains
        hmd_dict["protein"] = self.protein
        hmd_dict["sequence"] = self.sequence
        hmd_dict["topo_path"] = 'topo.pdb'
        hmd_dict["explicit_potential_code"] = inspect.getsource(HPS_potential)
        hmd_dict["explicit_pi_potential_code"] = inspect.getsource(HPS_pi_potential)
        hmd_dict["save"] = self.save
        hmd_dict["t"] = self.t
        hmd_dict["context"] = f"--gpu={self.n_gpus[np.where(self.temperatures==temp_K)[0][0]]}"
        hmd_dict["water_perm"] = self.water_perm
        hmd_dict["contract_t"] = self.contract_t
        hmd_dict["slab_t"] = self.slab_t
        hmd_dict["final_slab_x"] = round(self.slab_dimensions["x"], 2)/10.
        hmd_dict["final_slab_y"] = round(self.slab_dimensions["y"], 2)/10.
        hmd_dict["final_slab_z"] = round(self.slab_dimensions["z"], 2)/10.
        hmd_dict["hps_scale"] = self.hps_scale
        hmd_dict["debye"] = self.debye * 10.
        return hmd_dict

    def _generate_slurm(self, temp_K):
        """
        Generate a slm file to run at CSUC
        :return:
        """
        slurm_dict = {}
        if self.processors is not None:
            slurm_dict["command"] = f"python hmd_hps_{temp_K:.0f}.py"
            slurm_dict["tasktype"] = f"--ntasks={self.processors}"
            slurm_dict["queue"] = "std"
            slurm_dict["activate"] = "hoomd_cpu"
        else:
            slurm_dict["command"] = f"mpirun -np 1 python hmd_hps_{temp_K:.0f}.py  --mode=gpu"
            slurm_dict["tasktype"] = "--gres=gpu:1"
            slurm_dict["queue"] = "gpu"
            slurm_dict["activate"] = "hoomd_cpu"

        slurm_dict["jobname"] = self.job_name+f"-{temp_K:.0f}"
        return slurm_dict
