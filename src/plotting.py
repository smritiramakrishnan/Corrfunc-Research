import numpy as np
from Corrfunc.theory import wp
from Corrfunc.theory.DDrppi import DDrppi
import matplotlib.pyplot as plt
plt.style.use('MNRAS')

def get_frac_err(stdev, meanwp):
    err = [stdev[i]/meanwp[i] for i in range (len(stdev))]
    return err

def plot_fractional_error(error, wp, rpavg):
    fractional_error = [error[i]/wp[i] for i in range(len(error))]
    plt.plot(rpavg, fractional_error)
    plt.xscale('log')
    plt.title(r'$\sigma_{w_p}$/$w_p$ vs $\langle r_p\rangle$')
    plt.ylabel(r'$\sigma_{w_p}$/$w_p$')
    plt.xlabel(r'$\langle r_p\rangle$')

def plot_frac_err(rp_, frac_err, labels, colors, dotted = False, y_max = 0):
    for i in range(len(rp_)):
        if dotted:
            plt.plot(rp_[i], frac_err[i], color = colors[i], linestyle = '--')
        else:
            plt.plot(rp_[i], frac_err[i], color = colors[i])
    if y_max > 0:
        plt.ylim(0, y_max)
    plt.xscale('log')
    plt.title(r'$\sigma_{w_p}$/$w_p$ vs $\langle r_p\rangle$')
    plt.ylabel(r'$\sigma_{w_p}$/$w_p$')
    plt.xlabel(r'$\langle r_p\rangle$')
    plt.legend(labels, fontsize = 'small')

def wp_vs_rpavg(rpavg_, wp_, yerr, loglog = True):
    plt.plot(rpavg_, wp_, color = '#54186F')
    plt.errorbar(rpavg_, wp_, np.array(yerr), fmt = 'o',  capsize = 8, capthick = 1.5, color = "#54186F" , ecolor = "#A884BC", elinewidth = 1.5)
    plt.xlim(0.1, 21)
    plt.ylim(7, 4700)
    print(np.min(wp_), np.max(wp_))
    plt.title(r'$w_p$ vs $\langle r_p\rangle$')#, fontsize = 45)
    plt.ylabel(r'$w_p$')#, fontsize = 35)
    plt.xlabel(r'$\langle r_p\rangle$')#, fontsize = 35)
    if loglog:
        plt.loglog()
        
def error_v_rp(rpavg_, yerr, loglog = True):
    plt.plot(rpavg_, yerr)
    plt.title(r'$\langle r_p\rangle$ vs $\sigma_{w_p}$')
    plt.xlabel(r'$\langle r_p\rangle$')
    plt.ylabel(r'$\sigma_{w_p}$')
    if loglog:
        plt.loglog()


def plot_results(rpavg_, wp_, labels = '', error = []):
    for i in range(len(wp_)):
        plt.plot(rpavg_[i], wp_[i])
    plt.legend(labels, bbox_to_anchor= (1.5, 1), loc='upper right')
    plt.title("rpavg VS wp")
    plt.xlabel("rpavg")
    plt.ylabel("wp")
    if (len(error) > 0):
        plt.errorbar(rpavg_, wp_, yerr = error)
    plt.loglog()