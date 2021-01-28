import numpy as np
import scipy
from scipy.stats import nakagami, rayleigh
from scipy import constants


class Channel:
    def __init__(self):
        pass


class BANChannel(Channel):
    """Off-body Body Area Network Channel Class.

    Attributes
    ----------
    env: string
        `office`, `ferry`. Channel environment

    d0: float
        Some distance reference.


    Methods
    ----------
    step()
        Returns the BAN Channel loss.

    Reference
    ----------
    IRACON PROPAGATION MEASUREMENTS AND CHANNEL MODELS
    FOR 5G AND BEYOND, ITU Journal: ICT Discoveries,
    Vol. 2(1), 12 November 2019.
    """

    def __init__(self, env='office', d0=1, rnd=True):
        n_pl_dict = {
            'office': 1.71,
            'ferry': 1.69
        }
        L0_dict = {
            'office': 32.0,
            'ferry': 25.2
        }
        mu_dict = {
            'office': 0,
            'ferry': 0
        }
        sigma_dict = {
            'office': 1.2,
            'ferry': 1.7
        }
        m_dict = {
            'office': .9,
            'ferry': .8
        }
        omega_dict = {
            'office': 1.0,
            'ferry': 1.5
        }
        self.d0 = d0
        self.n_pl = n_pl_dict[env]
        self.L0 = L0_dict[env]
        self.mu = mu_dict[env]
        self.sigma = sigma_dict[env]
        self.m = m_dict[env]
        self.omega = omega_dict[env]
        self.rnd = rnd

    def step(self, d: float) -> float:
        """Returns the BAN Channel loss.

        Parameters
        ----------
        d : float
            Distance between TX and RX, in meters.

        Returns
        ----------
        loss: float
            Body Area Network Channel loss, in dB.

        """
        pathloss = self.pathloss(d)
        if self.rnd:
            large_scale_fading = self.large_scale()
            small_scale_fading = self.small_scale()
            loss = pathloss + large_scale_fading + small_scale_fading
        else:
            loss = pathloss
        return loss

    def pathloss(self, d: float) -> float:
        L0 = self.L0
        n_pl = self.n_pl
        d0 = self.d0
        # pathloss
        pathloss = L0 + 10*n_pl*np.log10(d/d0)
        # debugging
        # if pathloss == float('-inf'):
        #     print('bug')
        return pathloss

    def large_scale(self, *kargs) -> float:
        # large scale fading
        if self.rnd:
            large_scale_fading = np.random.normal(self.mu, self.sigma)
        else:
            large_scale_fading = 0
        return large_scale_fading

    def small_scale(self, *kargs) -> float:
        # small scale fading
        if self.rnd:
            nu = self.m / self.omega
            small_scale_fading = nakagami.rvs(nu)
        else:
            small_scale_fading = 0
        return small_scale_fading


class UrbanMacroLOSWinnerChannel:
    def __init__(self, rnd=True, h_bs=25, h_ms=1.5, f_c=2.4):
        self.rnd = rnd
        self.h_bs = h_bs
        self.h_ms = h_ms
        self.f_c = f_c

    def step(self, d: float) -> float:
        """Returns the line-of-sight urban macro-cell channel (C2)
        pathloss, according to the WINNER II Channel model.

        Parameters
        -----------
        d: float
            Distance between TX and RX, in meters.

        h_bs: float
            BS antenna height, in meters.

        h_ms: float
            User antenna height, in meters.

        f_c: float
            Carrier frequency, in GHz.

        Returns
        -----------
        loss: float
            Channel loss, in dB.

        loss_mag: float
            Channel loss, in magnitude.

        Reference
        -----------
        WINNER II Channel model.
        https://www.researchgate.net/publication/234055761_WINNER_II_channel_models
        """
        c = scipy.constants.c
        h_bs_eff = self.h_bs - 1
        h_ms_eff = self.h_ms - 1
        d_bp_eff = 4 * h_bs_eff * h_ms_eff * self.f_c / c
        if d > d_bp_eff and d < 5e3:
            loss = 40 * np.log10(d) + 13.47 - \
                14 * (np.log10(h_bs_eff) + np.log10(h_ms_eff)) + \
                6 * np.log10(self.f_c/5)
            sigma = 6
            # sigma = 1.2
            if self.rnd:
                large_scale_fading = np.random.normal(0, sigma)
                loss += large_scale_fading
            return loss
        elif d > 10 and d <= d_bp_eff:
            a = 26
            b = 39
            c = 20
            loss = a * np.log10(d) + b + c * np.log10(self.f_c/5)
            sigma = 4
            # sigma = 1.1
            if self.rnd:
                large_scale_fading = np.random.normal(0, sigma)
                loss += large_scale_fading
            return loss
        else:
            raise Exception('Invalid distance `d`.')

    def pathloss(self, d: float) -> float:
        c = scipy.constants.c
        h_bs_eff = self.h_bs - 1
        h_ms_eff = self.h_ms - 1
        d_bp_eff = 4 * h_bs_eff * h_ms_eff * self.f_c / c
        if d > d_bp_eff and d < 5e3:
            loss = 40 * np.log10(d) + 13.47 - \
                14 * (np.log10(h_bs_eff) + np.log10(h_ms_eff)) + \
                6 * np.log10(self.f_c/5)
            return loss
        elif d > 10 and d <= d_bp_eff:
            a = 26
            b = 39
            c = 20
            loss = a * np.log10(d) + b + c * np.log10(self.f_c/5)
            return loss

    def large_scale(self, d) -> float:
        if self.rnd:
            c = constants.c
            h_bs_eff = self.h_bs - 1
            h_ms_eff = self.h_ms - 1
            d_bp_eff = 4 * h_bs_eff * h_ms_eff * self.f_c / c
            if d > d_bp_eff and d < 5e3:
                sigma = 6
                large_scale_fading = np.random.normal(0, sigma)
                return large_scale_fading
            elif d > 10 and d <= d_bp_eff:
                sigma = 4
                large_scale_fading = np.random.normal(0, sigma)
                return large_scale_fading
            else:
                raise Exception('Invalid distance `d`.')
        else:
            return 0

    def small_scale(self, *kwargs) -> float:
        return 0


class UrbanMacroNLOSWinnerChannel:
    def __init__(self, rnd=True, h_bs=25, h_ms=1.5, f_c=2.4):
        self.rnd = rnd
        self.h_bs = h_bs
        self.h_ms = h_ms
        self.f_c = f_c
        self.sigma = 8

    def step(self, d: float) -> float:
        """Returns the no-line-of-sight urban macro-cell channel (C2)
        pathloss, according to the WINNER II Channel model.

        Parameters
        -----------
        d: float
            Distance between TX and RX, in meters.

        h_bs: float
            BS antenna height, in meters.

        h_ms: float
            User antenna height, in meters.

        f_c: float
            Carrier frequency, in GHz.

        Returns
        -----------
        loss: float
            Channel loss, in dB.

        loss_mag: float
            Channel loss, in magnitude.

        Reference
        -----------
        WINNER II Channel model.
        https://www.researchgate.net/publication/234055761_WINNER_II_channel_models
        """
        if d >= 5e3:
            raise Exception('Invalid distance `d`. It must be `d` < 5e3 meters.')  # noqa
        pathloss = self.pathloss(d)
        if self.rnd:
            loss = pathloss + self.large_scale(d) + self.small_scale()
        else:
            loss = pathloss
        return loss

    def pathloss(self, d: float) -> float:
        loss = (44.9-6.55*np.log10(self.h_bs))*np.log10(d) + \
            34.46+5.83*np.log10(self.h_bs)+23*np.log10(self.f_c/5)
        return loss

    def large_scale(self, d) -> float:
        loss = np.random.normal(0, self.sigma)
        return loss

    def small_scale(self, *kwargs) -> float:
        """Rayleigh distribution for the NLOS fading.
        """
        loss = rayleigh.rvs(scale=self.sigma, loc=0, size=1)
        return loss
