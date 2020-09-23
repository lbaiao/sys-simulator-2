from typing import Tuple
import numpy as np
import scipy
from scipy.stats import nakagami


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

    def step(self, d: float) -> Tuple[float, float]:
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
        L0 = self.L0
        n_pl = self.n_pl
        d0 = self.d0
        # pathloss
        pathloss = L0 + 10*n_pl*np.log10(d/d0)
        if self.rnd:
            # large scale fading
            large_scale_fading = np.random.lognormal(self.mu, self.sigma)
            # small scale fading
            nu = self.m / self.omega
            small_scale_fading = nakagami.rvs(nu)
            # total channel loss
            loss = pathloss + large_scale_fading + small_scale_fading
        else:
            loss = pathloss
        return loss


def UrbanMacroLOSWinnerChannel(
    d: float, h_bs: float, h_ms: float, f_c: float
) -> Tuple[float, float]:
    """Returns the line-of-sight urban macro-cell channel
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
    h_bs_eff = h_bs - 1
    h_ms_eff = h_ms - 1
    d_bp_eff = 4 * h_bs_eff * h_ms_eff * f_c / c
    if d > d_bp_eff and d < 5e3:
        loss = 40 * np.log10(d) + 13.47 - \
            14 * (np.log10(h_bs_eff) + np.log10(h_ms_eff)) + \
            6 * np.log10(f_c/5)
        return loss
    elif d > 10 and d <= d_bp_eff:
        a = 26
        b = 39
        c = 20
        loss = a * np.log10(d) + b + c * np.log10(f_c/5)
        return loss
    else:
        raise Exception('Invalid distance `d`.')
