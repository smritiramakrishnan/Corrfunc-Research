import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')

# returning fractional error of mean of multiple simulations
def get_frac_err(stdev, meanwp):
    err = [stdev[i]/meanwp[i] for i in range (len(stdev))]
    return err

def plot_frac_err(rp_, frac_err, labels, colors = None, dotted = False, y_max = 0, y_min = 0):
    for i in range(len(rp_)):
        if dotted:
            if colors != None:
                plt.plot(rp_[i], frac_err[i], color = colors[i], linestyle = '--', label = labels[i])
            else: 
                plt.plot(rp_[i], frac_err[i], linestyle = '--', label = labels[i])
        else:
            if colors != None:
                plt.plot(rp_[i], frac_err[i], color = colors[i], label = labels[i])
            else: 
                plt.plot(rp_[i], frac_err[i], label = labels[i])
    if y_max > 0:
        plt.ylim(y_min, y_max)
    plt.xscale('log')
    plt.title(r'$\sigma_{w_p}$/$w_p$ vs $\langle r_p\rangle$')
    plt.ylabel(r'$\sigma_{w_p}$/$w_p$')
    plt.xlabel(r'$\langle r_p\rangle$')
    plt.legend(labels, fontsize = 'small')

# plots wp and rp with error bars
def wp_vs_rpavg(rpavg_, wp_, yerr, loglog = True, y_max = 0, y_min = 0, legend = ''):
    #plt.plot(rpavg_, wp_, label = legend)
    plt.errorbar(rpavg_, wp_, yerr, label = legend)#, fmt = 'o',  capsize = 8, capthick = 1.5, color = "#54186F" , ecolor = "#A884BC", elinewidth = 1.5)
    if y_max > 0:
        plt.ylim(y_min, y_max)
    print(np.min(wp_), np.max(wp_))
    plt.title(r'$w_p$ vs $\langle r_p\rangle$')#, fontsize = 45)
    plt.ylabel(r'$w_p$')#, fontsize = 35)
    plt.xlabel(r'$\langle r_p\rangle$')#, fontsize = 35)
    if loglog:
        plt.loglog()

# plots stdev against rp 
def error_v_rp(rpavg_, yerr, loglog = True):
    plt.plot(rpavg_, yerr)
    plt.title(r'$\langle r_p\rangle$ vs $\sigma_{w_p}$')
    plt.xlabel(r'$\langle r_p\rangle$')
    plt.ylabel(r'$\sigma_{w_p}$')
    if loglog:
        plt.loglog()
        
def plot_results(rpavg_, wp_, labels = '', error = [], y_max = 0, y_min = 0):
    for i in range(len(wp_)):
        plt.plot(rpavg_[i], wp_[i])
    if y_max > 0:
        plt.ylim(y_min, y_max)
    plt.legend(labels, bbox_to_anchor= (1.5, 1), loc='upper right')
    plt.title(r'$w_p$ vs $\langle r_p\rangle$')
    plt.xlabel(r'$\langle r_p\rangle$')
    plt.ylabel(r'$w_p$')
    if (len(error) > 0):
        plt.errorbar(rpavg_, wp_, yerr = error)
    plt.loglog()