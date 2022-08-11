import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.typing import Union, Sequence, Optional, Dict
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from itertools import permutations, product
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from yaml import Loader
import warnings
import yaml
import sys

from emulator import Emulator
from predict_cl import AngularPowerSpectra

datavector_requires = {
    "p0": ["z_fid", "chiz_fid", "hz_fid"],
    "p2": ["z_fid", "chiz_fid", "hz_fid"],
    "p4": ["z_fid", "chiz_fid", "hz_fid"],
    "c_cmbkcmbk": [],
    "c_dcmbk": ["nz_d"],
    "c_kk": ["nz_s"],
    "c_dk": ["nz_d", "nz_s"],
    "c_dd": ["nz_d"],
}


class HarmonicSpaceWLxRSD(Likelihood):
    # From yaml file.
    heft: bool
    kecleft: bool
    datavector_info_filename: str
    zmin_proj: float
    zmax_proj: float
    nz_proj: int
    nz_pk: int
    z_pk: Optional[Union[Sequence, np.ndarray]]
    nchi_proj: int
    emulator_weights: Optional[Dict[str, str]]
    emulator_params: Optional[Dict[str, str]]
    use_lens_samples: Optional[Union[Sequence, np.ndarray]]
    use_source_samples: Optional[Union[Sequence, np.ndarray]]
    scale_cuts: Optional[Dict[str, float]]
    zstar: Optional[float]
    halofit_pmm: bool
    dummy_cov: bool

    # k = convergence
    # d = (galaxy) density
    # ignore scale cuts and windows for now

    def initialize(self):
        """Sets up the class."""
        self.z = np.linspace(self.zmin_proj, self.zmax_proj, self.nz_proj)

        if not hasattr(self, "z_pk"):
            self.z_pk = np.linspace(self.zmin_proj, self.zmax_proj, self.nz_pk)
        else:
            self.z_pk = np.array(self.z_pk)
            self.z_pk.sort()
            
        self.nk = 100
        self.k = np.logspace(-3, 0, self.nk)
        self.compute_c_kk = False
        self.compute_c_dk = False
        self.compute_c_dd = False
        self.compute_c_dcmbk = False
        self.compute_c_cmbkcmbk = False
        self.compute_p0 = False
        self.compute_p2 = False
        self.compute_p4 = False
        self.cell_emulators = False
        self.pell_emulators = False
        self.ndbins = 0
        self.nsbins = 0

        if not hasattr(self, "use_lens_samples"):
            self.use_lens_samples = None

        if not hasattr(self, "use_source_samples"):
            self.use_source_samples = None

        if not hasattr(self, "zstar"):
            self.zstar = 1098.0

        if not hasattr(self, "scale_cuts"):
            self.scale_cuts = None

        self.load_data()

        if not hasattr(self, "use_lens_samples"):
            self.use_lens_samples = np.arange(self.ndbins)

        if not hasattr(self, "use_source_samples"):
            self.use_source_samples = np.arange(self.nsbins)

        if self.compute_cell:
            self.setup_projection()

    def load_data(self):
        """Loads the required data."""

        with open(self.datavector_info_filename, "r") as fp:
            datavector_info = yaml.load(fp, Loader=Loader)

        # must specify a file with the actual correlation functions
        # in it. Should have the following columns: datavector type
        # (e.g. p0), redshift bin number for each sample being
        # correlated, separation values (e.g. k, ell, etc.), and
        # actual values for the spectra/correlation functions
        dt = np.dtype(
            [
                ("spectrum_type", "U10"),
                ("zbin0", np.int),
                ("zbin1", np.int),
                ("separation", np.float),
                ("value", np.float),
            ]
        )

        self.spectra = np.genfromtxt(
            datavector_info["spectra_filename"], names=True, dtype=dt
        )
        self.spectrum_types, idx = np.unique(self.spectra["spectrum_type"], return_index=True)
        #preserve order of spectra
        self.spectrum_types = self.spectrum_types[np.argsort(idx)]
        self.spectrum_info = {}

        self.compute_cell = False
        self.compute_pell = False

        if (
            ("c_kk" in self.spectrum_types)
            | ("c_dk" in self.spectrum_types)
            | ("c_dd" in self.spectrum_types)
            | ("c_cmbkcmbk" in self.spectrum_types)
            | ("c_dcmbk" in self.spectrum_types)
        ):
            self.compute_cell = True
        else:
            self.compute_cell = False

        if (
            ("p0" in self.spectrum_types)
            | ("p2" in self.spectrum_types)
            | ("p4" in self.spectrum_types)
        ):
            self.compute_pell = True
        else:
            self.compute_pell = False

        requirements = []
        for t in self.spectrum_types:
            # get rid of bins we don't want
            if (t[0] == "p") | (t[:3] == "c_d"):
                if self.use_lens_samples is not None:
                    idx = (
                        (self.spectra["spectrum_type"] == t)
                        & (np.in1d(self.spectra["zbin0"], self.use_lens_samples))
                    ) | (self.spectra["spectrum_type"] != t)

                    self.spectra = self.spectra[idx]

            if t[-1] == "k":
                if self.use_source_samples is not None:
                    idx = (
                        (self.spectra["spectrum_type"] == t)
                        & (np.in1d(self.spectra["zbin1"], self.use_source_samples))
                    ) | (self.spectra["spectrum_type"] != t)
                    self.spectra = self.spectra[idx]

            idx = self.spectra["spectrum_type"] == t
            setattr(self, "compute_{}".format(t), True)

            bins0 = np.unique(self.spectra[idx]["zbin0"])
            bins1 = np.unique(self.spectra[idx]["zbin1"])

            if (t[:3] == "c_d") | (t[0] == "p"):
                self.use_lens_samples = bins0
                self.ndbins = len(bins0)
            elif t[-1] == "k":
                self.use_source_samples = bins1
                self.nsbins = len(bins1)

            idx = self.spectra["spectrum_type"] == t
            z0 = self.spectra["zbin0"][idx][0]
            z1 = self.spectra["zbin1"][idx][0]
            idx &= (self.spectra["zbin0"] == z0) & (self.spectra["zbin1"] == z1)
            ndv_per_bin = np.sum(idx)
            sep_unmasked = self.spectra[idx]["separation"]

            self.spectrum_info[t] = {
                "bins0": bins0,
                "bins1": bins1,
                "n_dv_per_bin": ndv_per_bin,
                "separation": sep_unmasked,
            }

            requirements.extend(datavector_requires[t])

        self.n_dv = len(self.spectra)

        requirements = np.unique(np.array(requirements))

        for r in requirements:
            if r in ["z_fid", "chiz_fid", "hz_fid"]:
                setattr(self, r, datavector_info[r])
            else:
                setattr(
                    self, r, np.genfromtxt(datavector_info[r], dtype=None, names=True)
                )

        if "pkell_windows" in datavector_info.keys():
            window_matrix_files = datavector_info["pkell_windows"]
            assert "wide_angle_matrices" in datavector_info.keys()
            assert "kth_rsd" in datavector_info.keys()
            assert "ko_rsd" in datavector_info.keys()

            wide_angle_matrices = datavector_info["wide_angle_matrices"]
            kth_files = datavector_info["kth_rsd"]
            ko_files = datavector_info["ko_rsd"]

            self.Wmat = []
            self.M = []
            self.kth_rsd = []
            self.ko_rsd = []

            for i in range(len(window_matrix_files)):
                self.Wmat.append(np.loadtxt(window_matrix_files[i]))

            for i in range(len(wide_angle_matrices)):
                self.M.append(np.loadtxt(wide_angle_matrices[i]))

            for i in range(len(kth_files)):
                self.kth_rsd.append(np.loadtxt(kth_files[i]))

            for i in range(len(ko_files)):
                self.ko_rsd.append(np.loadtxt(ko_files[i]))

        if "cell_windows" in datavector_info.keys():
            window_matrix_files = datavector_info["cell_windows"]
#            assert "ell_obs" in datavector_info.keys()
#            ell_obs_files = datavector_info["ell_obs"]

            self.cW = {}
            for k in list(window_matrix_files.keys()):
                self.cW[k] = {}
                
                #want per bin windows here
                for ij in list(window_matrix_files[k].keys()):
                    self.cW[k][ij] = np.loadtxt(window_matrix_files[k][ij])

#            self.ell_obs = []
#            for i in range(len(ell_obs_files)):
#                self.ell_obs.append(np.loadtxt(ell_obs_files[i]))

        self.setup_scale_cuts()
        
        if not self.dummy_cov:
            self.load_covariance_matrix(datavector_info)
        else:
            self.cinv = None

        # check if we have emulators for these statistics
        if hasattr(self, "emulator_weights"):
            self.emulators = {}
            for stat in self.emulator_weights:
                # Only need one emulator per real space
                # stat.
                if stat in ["p_mm", "p_dm", "p_dd"]:
                    emu_path = self.emulator_weights[stat]
                    emu = Emulator(emu_path, kmax=1.0)
                    self.emulators[stat] = emu
                    self.cell_emulators = True

                elif stat in ["p0", "p2", "p4"]:
                    self.emulators[stat] = []
                    for i in self.spectrum_info[stat]["bins0"]:
                        emu_path = self.emulator_weights[stat][i]
                        emu = Emulator(emu_path, kmax=0.5)
                        self.emulators[stat].append(emu)
                    self.pell_emulators = True

    def setup_scale_cuts(self):
        # make scale cut mask
        if self.scale_cuts is not None:
            for t in self.spectrum_info:
                if t in self.scale_cuts:
                    scale_cut_dict = self.scale_cuts[t]
                    scale_cut_mask = {}
                    sep_unmasked = self.spectrum_info[t]["separation"]

                    if (t[0] == "p") | (t == "c_dd") | (t == "c_dcmbk"):
                        for i in self.use_lens_samples:
                            try:
                                if t == "c_dcmbk":
                                    sep_min, sep_max = scale_cut_dict[
                                        "{}_{}".format(i, 0)
                                    ]
                                    mask = (sep_min <= sep_unmasked) & (
                                        sep_unmasked <= sep_max
                                    )
                                    scale_cut_mask["{}_{}".format(i, 0)] = mask
                                else:
                                    sep_min, sep_max = scale_cut_dict[
                                        "{}_{}".format(i, i)
                                    ]
                                    mask = (sep_min <= sep_unmasked) & (
                                        sep_unmasked <= sep_max
                                    )
                                    scale_cut_mask["{}_{}".format(i, i)] = mask
                            except:
                                raise ValueError(
                                    "Scale cuts not provided for {} bin pair {},{}".format(
                                        t, i, i
                                    )
                                )

                    elif t == "c_dk":
                        for i in self.use_lens_samples:
                            for j in self.use_source_samples:
                                try:
                                    sep_min, sep_max = scale_cut_dict[
                                        "{}_{}".format(i, j)
                                    ]
                                except:
                                    raise ValueError(
                                        "Scale cuts not provided for {} bin pair {},{}".format(
                                            t, i, j
                                        )
                                    )

                                mask = (sep_min <= sep_unmasked) & (
                                    sep_unmasked <= sep_max
                                )
                                scale_cut_mask["{}_{}".format(i, j)] = mask

                    elif t == "c_kk":
                        for i in self.use_source_samples:
                            for j in self.use_source_samples:
                                try:
                                    sep_min, sep_max = scale_cut_dict[
                                        "{}_{}".format(i, j)
                                    ]
                                except:
                                    raise ValueError(
                                        "Scale cuts not provided for {} bin pair {},{}".format(
                                            t, i, j
                                        )
                                    )

                                mask = (sep_min <= sep_unmasked) & (
                                    sep_unmasked <= sep_max
                                )
                                scale_cut_mask["{}_{}".format(i, j)] = mask

                    elif t == "c_cmbkcmbk":
                        try:
                            sep_min, sep_max = scale_cut_dict["0_0"]
                        except:
                            raise ValueError(
                                "Scale cuts not provided for {} bin pair {},{}".format(
                                    t, 0, 0
                                )
                            )

                        mask = (sep_min <= sep_unmasked) & (sep_unmasked <= sep_max)
                        scale_cut_mask["0_0"] = mask

                    self.spectrum_info[t]["scale_cut_masks"] = scale_cut_mask

                else:
                    raise ValueError("No scale cuts specified for {}".format(t))
        else:
            warnings.warn("No scale cuts specified for any spectra!", UserWarning)

            for t in self.spectrum_info:
                self.spectrum_info[t]["scale_cut_masks"] = None

    def load_covariance_matrix(self, datavector_info):
        # Always need a covariance matrix. This should be a text file
        # with columns specifying the two data vector types, and four redshift
        # bin indices for each element, as well as a column for the elements
        # themselves
        dt = np.dtype(
            [
                ("spectrum_type0", "U10"),
                ("spectrum_type1", "U10"),
                ("zbin00", np.int),
                ("zbin01", np.int),
                ("zbin10", np.int),
                ("zbin11", np.int),
                ("separation0", np.float),
                ("separation1", np.float),
                ("value", np.float),
            ]
        )

        cov_raw = np.genfromtxt(
            datavector_info["covariance_filename"], names=True, dtype=dt
        )

        self.cov = np.zeros((self.n_dv, self.n_dv))

        spec_type_counter = []
        zbin_counter = {}
        cov_scale_mask = []

        for t0, t1 in product(self.spectrum_types, self.spectrum_types):
            if (t0, t1) in spec_type_counter:
                continue

            if (t0, t1) not in zbin_counter.keys():
                zbin_counter[(t0, t1)] = []

            n00, n01, n10, n11 = (
                self.spectrum_info[t0]["bins0"],
                self.spectrum_info[t0]["bins1"],
                self.spectrum_info[t1]["bins0"],
                self.spectrum_info[t1]["bins1"],
            )

            for zb00, zb01, zb10, zb11 in product(n00, n01, n10, n11):
                if ((zb00, zb01), (zb10, zb11)) in zbin_counter[(t0, t1)]:
                    continue

                idxi = np.where(
                    (self.spectra["spectrum_type"] == t0)
                    & (self.spectra["zbin0"] == zb00)
                    & (self.spectra["zbin1"] == zb01)
                )[0]

                idxj = np.where(
                    (self.spectra["spectrum_type"] == t1)
                    & (self.spectra["zbin0"] == zb10)
                    & (self.spectra["zbin1"] == zb11)
                )[0]

                covidx = (
                    (cov_raw["spectrum_type0"] == t0)
                    & (cov_raw["zbin00"] == zb00)
                    & (cov_raw["zbin01"] == zb01)
                    & (cov_raw["spectrum_type1"] == t1)
                    & (cov_raw["zbin10"] == zb10)
                    & (cov_raw["zbin11"] == zb11)
                )

                if len(idxi) == 0:
                    #                    print("Warning: didn't find any spectra with spec_type={}, zb0={}, zb1={}. Proceeding, but double check your data vector and covariance".format(t0, zb00, zb01))
                    zbin_counter[(t0, t1)].extend(
                        permutations(((zb00, zb01), (zb10, zb11)))
                    )

                    continue
                if len(idxj) == 0:
                    #                    print("Warning: didn't find any spectra with spec_type={}, zb0={}, zb1={}. Proceeding, but double check your data vector and covariance".format(t1, zb10, zb11))
                    zbin_counter[(t0, t1)].extend(
                        permutations(((zb00, zb01), (zb10, zb11)))
                    )
                    continue

                ii, jj = np.meshgrid(idxi, idxj, indexing="ij")

                start_idx = np.min(idxi)
                # mask scales
                if self.spectrum_info[t0]["scale_cut_masks"] is not None:
                    mask_i = np.where(
                        self.spectrum_info[t0]["scale_cut_masks"][
                            "{}_{}".format(zb00, zb01)
                        ]
                    )[0]
                else:
                    mask_i = np.arange(self.spectrum_info[t0]["n_dv_per_bin"])

                cov_scale_mask.extend((mask_i + start_idx).tolist())

                if np.sum(covidx) > 0:
                    self.cov[ii.flatten(), jj.flatten()] = cov_raw[covidx]["value"]

                zbin_counter[(t0, t1)].extend(
                    permutations(((zb00, zb01), (zb10, zb11)))
                )

            spec_type_counter.extend(permutations((t0, t1)))

        # reflect triangular matrix to give symmetric cov
        self.cov += self.cov.T
        self.cov[np.arange(self.n_dv), np.arange(self.n_dv)] /= 2
        self.scale_mask = np.unique(cov_scale_mask)
        self.scale_mask.sort()
        self.n_dv_masked = len(self.scale_mask)
        cov_scale_mask_i, cov_scale_mask_j = np.meshgrid(
            self.scale_mask, self.scale_mask, indexing="ij"
        )

        self.cinv = np.linalg.inv(
            self.cov[cov_scale_mask_i, cov_scale_mask_j].reshape(
                self.n_dv_masked, self.n_dv_masked
            )
        )
        print("done load cov", flush=True)

    def setup_projection(self):

        if hasattr(self, "nz_d"):
            dndz_lens = [
                Spline(self.nz_d["z"], self.nz_d["nz{}".format(i)], ext=1)(self.z)
                for i in self.use_lens_samples
            ]
        else:
            dndz_lens = None

        if hasattr(self, "nz_s"):
            dndz_source = [
                Spline(self.nz_s["z"], self.nz_s["nz{}".format(i)], ext=1)(self.z)
                for i in self.use_source_samples
            ]
        else:
            dndz_source = None

        compute_d_x_kcmb = "c_dcmbk" in list(self.spectrum_info.keys())
        compute_kcmb_x_kcmb = "c_cmbkcmbk" in list(self.spectrum_info.keys())

        self.projection_model = AngularPowerSpectra(
            self.z,
            dndz_lens,
            dndz_source,
            d_x_cmbk=compute_d_x_kcmb,
            cmbk_x_cmbk=compute_kcmb_x_kcmb,
        )

    def get_requirements(self):

        reqs = {}
        if self.compute_cell:
            if not self.cell_emulators:
                if self.heft:
                    reqs.update(
                        {
                            "heft_spectrum_interpolator": {"z": self.z_pk},
                            "heft_spectrum_grid": {"z": self.z_pk},
                        }
                    )
                else:
                    reqs.update({"eft_spectrum_interpolator": {"z": self.z_pk}})
            else:
                reqs.update(
                    {
                        "logA": None,
                        "ns": None,
                        "H0": None,
                        "w": None,
                        "ombh2": None,
                        "omch2": None,
                    }
                )

            z = np.concatenate([self.z, [1098.0]])
            reqs.update(
                {
                    "comoving_radial_distance": {"z": z},
                    "Hubble": {"z": self.z},
                    "H0": None,
                    "omegam": None,
                }
            )
            if self.halofit_pmm:
                reqs.update(
                    {"Pk_interpolator": {"k_max": 10, "z": self.z_pk, "nonlinear": True}}
                )

            reqs.update({"sigma8_z": {"z": self.z_pk}})

        if self.compute_pell:
            if not self.pell_emulators:
                reqs["pt_pk_ell_model"] = {}
                reqs["pt_pk_ell_model"]["z"] = self.z_fid
                reqs["pt_pk_ell_model"]["chiz_fid"] = self.chiz_fid
                reqs["pt_pk_ell_model"]["hz_fid"] = self.hz_fid
            else:
                reqs.update(
                    {
                        "logA": None,
                        "ns": None,
                        "H0": None,
                        "w": None,
                        "ombh2": None,
                        "omch2": None,
                    }
                )

        return reqs

    def combine_real_space_spectra(self, k, spectra, bias_params, cross=False):

        pkvec = np.zeros((14, spectra.shape[1], spectra.shape[2]))
        pkvec[:10, ...] = spectra
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]

        # Higher derivative terms
        pkvec[10:, ...] = -k[np.newaxis, :, np.newaxis] ** 2 * pkvec[nabla_idx, ...]

        b1, b2, bs, bk2, sn = bias_params
        if not cross:
            bterms = [
                1,
                2 * b1,
                b1**2,
                b2,
                b2 * b1,
                0.25 * b2**2,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2,
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]
        else:
            # hm correlations only have one kind of <1,delta_i> correlation
            bterms = [1, b1, 0, b2 / 2, 0, 0, bs, 0, 0, 0, bk2, 0, 0, 0]

        p = np.einsum("b, bkz->kz", bterms, pkvec)

        if not cross:
            p += sn

        return p

    def combine_cleft_bias_terms_pk(self, kv, arr, b1, b2, bs, b3, alpha, sn):
        """
        Combine all the bias terms into one power spectrum,
        where alpha is the counterterm and sn the shot noise/stochastic contribution.
        """
        pkarr = np.copy(arr)

        pkarr[1, :] *= 2
        pkarr[5, :] *= 0.25
        pkarr[6, :] *= 2
        pkarr[7, :] *= 2
        bias_monomials = np.array(
            [
                1,
                b1,
                b1**2,
                b2,
                b1 * b2,
                b2**2,
                bs,
                b1 * bs,
                b2 * bs,
                bs**2,
                b3,
                b1 * b3,
            ]
        )
        za = pkarr[-1, :]
        pktemp = np.copy(pkarr)[:-1, ...]

        res = (
            np.sum(pktemp * bias_monomials[:, np.newaxis, np.newaxis], axis=0)
            + alpha * kv[:, np.newaxis] ** 2 * za
            + sn
        )

        return res

    def combine_kecleft_bias_terms_pk(self, kv, arr, b1, b2, bs, b3, alpha, sn):
        """
        Combine all the bias terms into one power spectrum,
        where alpha is the counterterm and sn the shot noise/stochastic contribution.
        """
        pkarr = np.copy(arr)

        pkarr[1, :] *= 2
        pkarr[5, :] *= 0.25
        pkarr[6, :] *= 2
        pkarr[7, :] *= 2
        bias_monomials = np.array(
            [1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2]
        )
        pktemp = np.copy(pkarr)

        res = np.sum(pktemp * bias_monomials[:, np.newaxis, np.newaxis], axis=0) + sn

        return res

    def combine_cleft_bias_terms_pk_crossmatter(self, kv, arr, b1, b2, bs, b3, alpha):
        """A helper function to return P_{gm}, which is a common use-case."""
        pkarr = np.copy(arr)
        pkarr[1, :] *= 2
        pkarr[5, :] *= 0.25
        pkarr[6, :] *= 2
        pkarr[7, :] *= 2
        ret = (
            pkarr[0, :]
            + 0.5 * b1 * pkarr[1, :]
            + 0.5 * b2 * pkarr[3, :]
            + 0.5 * bs * pkarr[6, :]
            + 0.5 * b3 * pkarr[10, :]
            + alpha * kv[:, np.newaxis] ** 2 * pkarr[12, :]
        )
        return ret

    def combine_kecleft_bias_terms_pk_crossmatter(self, kv, arr, b1, b2, bs, b3, alpha):
        """A helper function to return P_{gm}, which is a common use-case."""
        pkarr = np.copy(arr)
        pkarr[1, :] *= 2
        pkarr[5, :] *= 0.25
        pkarr[6, :] *= 2
        pkarr[7, :] *= 2
        ret = (
            pkarr[0, :]
            + 0.5 * b1 * pkarr[1, :]
            + 0.5 * b2 * pkarr[3, :]
            + 0.5 * bs * pkarr[6, :]
        )
        return ret

    def combine_pkell_spectra(self, bvec, pktable):
        """
        Returns pell, assuming AP parameters from input p{ell}ktable
        """

        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = bvec
        bias_monomials = np.array(
            [
                1,
                b1,
                b1**2,
                b2,
                b1 * b2,
                b2**2,
                bs,
                b1 * bs,
                b2 * bs,
                bs**2,
                b3,
                b1 * b3,
                alpha0,
                alpha2,
                alpha4,
                alpha6,
                sn,
                sn2,
                sn4,
            ]
        )
        p = np.sum(pktable * bias_monomials[:, np.newaxis, np.newaxis], axis=0)

        return p

    def observe_theory_nowindow(self, theory, sep_th, sep_obs):
        """Bin into k bins."""

        rdat = sep_obs
        dr = rdat[1] - rdat[0]
        nr = len(rdat)

        thy = interp1d(sep_th, theory, fill_value="extrapolate", axis=0, kind="cubic")
        thy_obs = np.zeros((nr, theory.shape[1]))

        for i in range(rdat.size):
            kl = rdat[i] - dr / 2
            kr = rdat[i] + dr / 2

            ss = np.linspace(kl, kr, 100)
            p0 = thy(ss)
            thy_obs[i] = (
                np.trapz(ss[:, np.newaxis] ** 2 * p0, x=ss, axis=0)
                * 3
                / (kr**3 - kl**3)
            )

        return thy_obs

    def observe_theory_wide_angle_and_window(
            self, p0_theory, p2_theory, p4_theory, M, W, kth, ko
    ):

        nobs = len(ko)
        p0interp = interp1d(self.k_rsd, p0_theory, fill_value="extrapolate")
        p2interp = interp1d(self.k_rsd, p2_theory, fill_value="extrapolate")
        p4interp = interp1d(self.k_rsd, p4_theory, fill_value="extrapolate")

        p0model = p0interp(kth)
        p2model = p2interp(kth)
        p4model = p4interp(kth)
        model_conv = np.dot(W, np.dot(M, np.hstack([p0model, p2model, p4model])))

        return model_conv[:nobs], model_conv[nobs:2*nobs], model_conv[2*nobs:3*nobs], model_conv[3*nobs:4*nobs], model_conv[4*nobs:]

    def observe_cell(self, cell_theory, W, spectrum_type, Lmax=1001):
        
        n_ell_obs = W[list(W.keys())[0]].shape[0]
        cell_obs = np.zeros((n_ell_obs, cell_theory.shape[-1]))
        counter = 0
        
        if spectrum_type == 'c_cmbkcmbk':
            cell_obs[:,0] = np.dot(W['0_0'][:,:Lmax], cell_theory[:Lmax, 0])
        
        elif spectrum_type == 'c_kk':

            for i in self.use_source_samples:
                for j in self.use_source_samples:
                    if j<i:continue
                    cell_obs[:,counter] = np.dot(W['{}_{}'.format(i,j)][:,:Lmax], cell_theory[:Lmax, counter])
                    counter += 1
        
        elif spectrum_type == 'c_dk':
            
            for i in self.use_lens_samples:
                for j in self.use_source_samples:
                    cell_obs[:,counter] = np.dot(W['{}_{}'.format(i,j)][:,:Lmax], cell_theory[:Lmax, counter])
                    counter += 1 

        elif (spectrum_type == 'c_dcmbk') | (spectrum_type == 'c_dd'):
            for i in self.use_lens_samples:
                cell_obs[:,counter] = np.dot(W['{}'.format(i)][:,:Lmax], cell_theory[:Lmax, counter])
                counter += 1
                
        return cell_obs


    def logp(self, **params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""
        pmm_spline = None
        pdm_splines = []
        pdd_splines = []

        if self.compute_cell:
            h = self.provider.get_param("H0") / 100
            z = np.concatenate([self.z, [1098.0]])
            chiz = self.provider.get_comoving_radial_distance(z) * h
            chistar = chiz[-1]
            chiz = chiz[:-1]
            omega_m = self.provider.get_param("omegam")
            sigma8_z = self.provider.get_sigma8_z(self.z_pk)
            sigma8_z_spl = interp1d(self.z_pk, sigma8_z, fill_value='extrapolate', kind='cubic')
            sigma8_z = sigma8_z_spl(self.z)
            Dz = sigma8_z / sigma8_z[0]

            ez = self.provider.get_Hubble(self.z) / self.provider.get_param("H0")

            if not self.cell_emulators:
                if self.heft:
                    cell_spec_interpolator = self.provider.get_result(
                        "heft_spectrum_interpolator"
                    )
                else:
                    cell_spec_interpolator = self.provider.get_result(
                        "eft_spectrum_interpolator"
                    )

                if self.halofit_pmm:
                    pmm_interpolator = self.provider.get_Pk_interpolator(nonlinear=True)

                n_spec_cell = len(cell_spec_interpolator)
                self.rs_power_spectra = np.zeros(
                    (self.ndbins, self.nz_pk, self.nk, 3)
                )
            else:
                cosmo_params = [
                    self.provider.get_param(p)
                    for p in ["logA", "ns", "H0", "w", "ombh2", "omch2"]
                ]

                self.rs_power_spectra = np.zeros(
                    (self.ndbins, self.nz_pk, self.emulators["p_mm"].nk, 3)
                )

            for idx, i in enumerate(self.use_lens_samples):

                b1 = params_values["b1_{}".format(i)]
                b2 = params_values["b2_{}".format(i)]
                bs = params_values["bs_{}".format(i)]
                bk = params_values["bk_{}".format(i)]
                sn = params_values["sn_{}".format(i)]
                bias_params = [b1, b2, bs, bk, sn]
                if not self.cell_emulators:

                    spectra = np.zeros((n_spec_cell, self.nk, self.nz_pk))
                    for j in range(n_spec_cell):
                        if (j == 0) & self.halofit_pmm:
                            spectra[j, ...] = (
                                pmm_interpolator.P(self.z_pk, self.k * h).T * h**3
                            )
                        else:
                            spectra[j, ...] = (
                                cell_spec_interpolator[j].P(self.z_pk, self.k).T
                            )

                    if self.heft:
                        pmm = spectra[0, ...]
                        pdm = self.combine_real_space_spectra(
                            self.k, spectra, bias_params, cross=True
                        )
                        pdd = self.combine_real_space_spectra(
                            self.k, spectra, bias_params, cross=False
                        )
                        k = self.k
                    else:
                        if self.halofit_pmm:
                            pmm = spectra[0, ...]
                        elif self.kecleft:
                            if bk != 0:
                                warnings.warn(
                                    "b_k!=0, but you are using kecleft, so counterterm is not computed.",
                                    UserWarning,
                                )
                            pmm = self.combine_kecleft_bias_terms_pk(
                                self.k, spectra, 0, 0, 0, 0, bk, 0
                            )
                        else:
                            pmm = self.combine_cleft_bias_terms_pk(
                                self.k, spectra, 0, 0, 0, 0, bk, 0
                            )
                        if self.kecleft:
                            if bk != 0:
                                warnings.warn(
                                    "b_k!=0, but you are using kecleft, so counterterm is not computed.",
                                    UserWarning,
                                )
                            pdm = self.combine_kecleft_bias_terms_pk_crossmatter(
                                self.k, spectra, b1, b2, bs, 0.0, bk
                            )
                            pdd = self.combine_kecleft_bias_terms_pk(
                                self.k, spectra, b1, b2, bs, 0.0, bk, sn
                            )
                        else:
                            pdm = self.combine_cleft_bias_terms_pk_crossmatter(
                                self.k, spectra, b1, b2, bs, 0.0, bk
                            )
                            pdd = self.combine_cleft_bias_terms_pk(
                                self.k, spectra, b1, b2, bs, 0.0, bk, sn
                            )

                else:
                    params = cosmo_params + bias_params

                    cparam_grid = np.zeros((len(self.z_pk), len(cosmo_params) + 1))
                    param_grid = np.zeros((len(self.z_pk), len(params) + 1))

                    params = np.array(params)
                    cosmo_params = np.array(cosmo_params)

                    cparam_grid[:, :-1] = cosmo_params
                    param_grid[:, :-1] = params
                    cparam_grid[:, -1] = self.z_pk
                    param_grid[:, -1] = self.z_pk

                    k, pmm = self.emulators["p_mm"](cparam_grid)
                    k, pdm = self.emulators["p_dm"](param_grid)
                    k, pdd = self.emulators["p_dd"](param_grid)

                    pmm = pmm.T
                    pdm = pdm.T
                    pdd = pdd.T

                pmm_spline = PowerSpectrumInterpolator(self.z_pk, self.k, pmm.T)
                self.rs_power_spectra[idx, :, :, 0] = pmm.T

                pdm_spline = PowerSpectrumInterpolator(self.z_pk, self.k, pdm.T)
                self.rs_power_spectra[idx, :, :, 1] = pdm.T

                pdd_spline = PowerSpectrumInterpolator(self.z_pk, self.k, pdd.T)
                self.rs_power_spectra[idx, :, :, 2] = pdd.T

                pmm_spline = pmm_spline
                pdm_splines.append(pdm_spline)
                pdd_splines.append(pdd_spline)

            smag = np.array(
                [params_values["smag_{}".format(d)] for d in self.use_lens_samples]
            )
            a_ia = params_values["a_ia"]
            eta_ia = params_values["eta_ia"]

            lval, Cdd, Cdk, Ckk, Cdcmbk, Ccmbkcmbk = self.projection_model(
                pmm_spline,
                pdm_splines,
                pdd_splines,
                chiz,
                ez,
                omega_m,
                chistar,
                Dz,
                smag,
                a_ia,
                eta_ia,
            )

            # just interpolate onto desired ell for now.

            if self.compute_c_kk:
                
                if hasattr(self, 'cW'):
                    Ckk_eval = self.observe_cell(
                              Ckk, self.cW['c_kk'], 'c_kk'
                                )
                else:
                    Ckk_eval = self.observe_theory_nowindow(
                    Ckk, lval, self.spectrum_info["c_kk"]["separation"]
                )
                counter = 0
                for i in self.use_source_samples:
                    for j in self.use_source_samples:
                        if j>i:continue
                        self.spectrum_info["c_kk"][
                            "{}_{}_model".format(i, j)
                        ] = Ckk_eval[:, counter]
                        self.spectrum_info["c_kk"][
                            "{}_{}_model_unobs".format(i, j)
                        ] = Ckk[:, counter]                        
                        counter += 1

            if self.compute_c_dk:
                
                if hasattr(self, 'cW'):
                    Cdk_eval = self.observe_cell(Cdk, self.cW['c_dk'], 'c_dk')
                else:
                    Cdk_eval = self.observe_theory_nowindow(
                    Cdk, lval, self.spectrum_info["c_dk"]["separation"]
                    )
                
                counter = 0
                for i in self.use_lens_samples:
                    for j in self.use_source_samples:
                        self.spectrum_info["c_dk"][
                            "{}_{}_model".format(i, j)
                        ] = Cdk_eval[:, counter]
                        
                        self.spectrum_info["c_dk"][
                            "{}_{}_model_unobs".format(i, j)
                        ] = Cdk[:, counter]                        
                        counter += 1

            if self.compute_c_dd:
                if hasattr(self, 'cW'):
                    Cdd_eval = self.observe_cell(Cdd, self.cW['c_dd'], 'c_dd')
                else:
                    Cdd_eval = self.observe_theory_nowindow(
                        Cdd, lval, self.spectrum_info["c_dd"]["separation"]
                    )
                counter = 0
                for i in self.use_lens_samples:
                    self.spectrum_info["c_dd"]["{}_model".format(i)] = Cdd_eval[:, counter]
                    self.spectrum_info["c_dd"]["{}_model_unobs".format(i)] = Cdd[:, counter]
                    
                    counter += 1

            if self.compute_c_dcmbk:
                if hasattr(self, 'cW'):
                    Cdcmbk_eval = self.observe_cell(Cdcmbk, self.cW['c_dcmbk'], 'c_dcmbk')
                else:
                    Cdcmbk_eval = self.observe_theory_nowindow(
                        Cdcmbk, lval, self.spectrum_info["c_dcmbk"]["separation"]
                    )
                    
                counter = 0
                for i in self.use_lens_samples:
                    self.spectrum_info["c_dcmbk"]["{}_model".format(i)] = Cdcmbk_eval[
                        :, counter
                    ]
                    self.spectrum_info["c_dcmbk"]["{}_model_unobs".format(i)] = Cdcmbk[
                        :, counter
                    ]                    
                    counter += 1

            if self.compute_c_cmbkcmbk:
                if hasattr(self, 'cW'):
                    Ccmbkcmbk_eval = self.observe_cell(Ccmbkcmbk[:,np.newaxis], self.cW['c_cmbkcmbk'], 'c_cmbkcmbk')
                else:
                    Ccmbkcmbk_eval = self.observe_theory_nowindow(
                        Ccmbkcmbk[:, np.newaxis],
                        lval,
                        self.spectrum_info["c_cmbkcmbk"]["separation"],
                    )
                    
                self.spectrum_info["c_cmbkcmbk"]["model"] = Ccmbkcmbk_eval[:, 0]
                self.spectrum_info["c_cmbkcmbk"]["model_unobs"] = Ccmbkcmbk

        if self.compute_pell:

            if not self.pell_emulators:
                pt_models = self.provider.get_result("pt_pk_ell_model")
            else:
                # all emulators should take cosmo params in the same order
                cosmo_params = self.emulator_params["p0"]["cosmology"]
                cosmo_params = [self.provider.get_param(p) for p in cosmo_params]
            for idx, i in enumerate(self.use_lens_samples):
                # compute multipoles and interpolate onto desired k values
                # need to implement window/wide angle stuff still.
                if not self.pell_emulators:
                    b1 = params_values["b1_{}".format(i)]
                    b2 = params_values["b2_{}".format(i)]
                    bs = params_values["bs_{}".format(i)]
                    b3 = params_values["b3_{}".format(i)]

                    alpha0 = params_values["alpha0_{}".format(i)]
                    alpha2 = params_values["alpha2_{}".format(i)]
                    alpha4 = params_values["alpha4_{}".format(i)]
                    alpha6 = params_values["alpha6_{}".format(i)]

                    sn = params_values["sn_{}".format(i)]
                    sn2 = params_values["sn2_{}".format(i)]
                    sn4 = params_values["sn4_{}".format(i)]

                    bias_params = [
                        b1,
                        b2,
                        bs,
                        b3,
                        alpha0,
                        alpha2,
                        alpha4,
                        alpha6,
                        sn,
                        sn2,
                        sn4,
                    ]

                    lpt = pt_models[i]
                    k, p0, p2, p4 = lpt.combine_bias_terms_pkell(bias_params)
                    self.k_rsd = k
                else:

                    bias_params = self.emulator_params["p0"]["bias"]
                    bias_params = [
                        params_values["{}_{}".format(p, i)] for p in bias_params
                    ]

                    params = cosmo_params + bias_params

                    params = np.array(params)

                    if self.compute_p0:
                        k, p0 = self.emulators["p0"][idx](params)
                    if self.compute_p2:
                        k, p2 = self.emulators["p2"][idx](params)
                    if self.compute_p4:
                        k, p4 = self.emulators["p4"][idx](params)

                    self.k_rsd = k

                if not hasattr(self, "pkell_spectra"):
                    self.pkell_spectra = np.zeros((self.ndbins, len(k), 3))

                if self.compute_p0:
                    self.pkell_spectra[idx, :, 0] = p0
                if self.compute_p2:
                    self.pkell_spectra[idx, :, 1] = p2
                if self.compute_p4:
                    self.pkell_spectra[idx, :, 2] = p4

                if hasattr(self, "Wmat"):
                    p0_obs, _, p2_obs, _, p4_obs = self.observe_theory_wide_angle_and_window(
                        p0,  p2, p4, self.M[idx], self.Wmat[idx], self.kth_rsd[idx],
                        self.ko_rsd[idx]
                    )
                    self.spectrum_info["p0"]["{}_model".format(i)] = p0_obs
                    self.spectrum_info["p2"]["{}_model".format(i)] = p2_obs
                    self.spectrum_info["p4"]["{}_model".format(i)] = p4_obs
                else:
                    if self.compute_p0:
                        p0_obs = self.observe_theory_nowindow(
                            p0[:, np.newaxis], k, self.spectrum_info["p0"]["separation"]
                        )
                        self.spectrum_info["p0"]["{}_model".format(i)] = p0_obs[:, 0]

                    if self.compute_p2:
                        p2_obs = self.observe_theory_nowindow(
                            p2[:, np.newaxis], k, self.spectrum_info["p2"]["separation"]
                        )
                        self.spectrum_info["p2"]["{}_model".format(i)] = p2_obs[:, 0]

                    if self.compute_p4:
                        p4_obs = self.observe_theory_nowindow(
                            p4[:, np.newaxis], k, self.spectrum_info["p4"]["separation"]
                        )
                        self.spectrum_info["p4"]["{}_model".format(i)] = p4_obs[:, 0]

        # package everything up into one big model datavector
        model = []
        for t in self.spectrum_types:
            if t[0] == "p":
                for i in self.use_lens_samples:
                    m = self.spectrum_info[t]["{}_model".format(i)]
                    model.append(m)
            elif t[2] == "k":
                for i in self.use_source_samples:
                    for j in self.use_source_samples:
                        m = self.spectrum_info[t]["{}_{}_model".format(i, j)]
                        model.append(m)

            elif (t[2] == "d") & (t[3] == "k"):
                for i in self.use_lens_samples:
                    for j in self.use_source_samples:
                        m = self.spectrum_info[t]["{}_{}_model".format(i, j)]
                        model.append(m)

            elif (t[3] == "d") | (t[3] == "c"):
                for i in self.use_lens_samples:
                    m = self.spectrum_info[t]["{}_model".format(i)]
                    model.append(m)

            else:
                m = self.spectrum_info[t]["model".format(i)]
                model.append(m)

        model = np.hstack(model)
        self.model_pred = model

        if not self.dummy_cov:
            mask = self.scale_mask
            diff = self.spectra["value"][mask] - model[mask]
            chi2 = np.dot(diff, np.dot(self.cinv, diff))
        else:
            chi2 = 1
            
        return -0.5 * chi2
