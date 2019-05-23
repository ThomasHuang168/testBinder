import numpy as np
import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    
import matplotlib as mpl
import matplotlib.pyplot as plt

from ipywidgets import interact, widgets, Layout
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from IPython import display
from IPython.display import Javascript

from other_func import find_gt_in_log, find_top_limit, find_threshold
import os

# Define the random seed for experiment
np.random.seed(0)
torch.manual_seed(0)

# Define parameters to be selected for viewing of the experiment results
sample_sizes = [400]
batch_sizes_temp = [0.25*sample_sizes[0], 0.5*sample_sizes[0]]
batch_sizes = [int(bs) for bs in batch_sizes_temp]
rhos = [0.4, 0.6, 0.8, 0.9]
hidden_layers = [300]

# Define dropdown menus
sample_sizes_drop = widgets.Dropdown(options=sample_sizes, description="Sample size:")
batch_sizes_drop = widgets.Dropdown(options=batch_sizes, description="Batch size:")
rhos_drop = widgets.Dropdown(options=rhos, description="Rho:", value=0.9)
ma_rate_box = widgets.FloatText(value=0.01, description="Moving average rate:", disabled=False)

# List available dropdown menus for easier rendering
widget_list = [sample_sizes_drop, batch_sizes_drop, rhos_drop, ma_rate_box]

# Define output box to display results
result_out = widgets.Output()

def run_exp(ev):
    # Event handler to show result on button press
    global result_out
    with result_out:
        display.clear_output()
        show_result()

# Button to show result
run_btn = widgets.Button(description="View result")
run_btn.style.button_color = "lightgreen"
run_btn.on_click(run_exp)

def show_widgets():
    # Render dropdown menus and button
    global widget_list
    global run_btn
    for widg in widget_list:
        widg.style = {'description_width': '200px'}
        widg.layout=dict(width='400px')
        display.display(widg)
    display.display(run_btn)

# Initialize display
show_widgets()
display.display(result_out)

def show_result():
    # Show results according to selected parameters
    global widget_list

    # Initialize variables according to parameters selected by user
    sample_size = sample_sizes_drop.value
    batch_size = int(batch_sizes_drop.value)
    rho = rhos_drop.value
    iteration = 25600

    # Define folder name to load from
    folder_name_mine = ("pop=%s_batch=%s_Mixed Gaussian_rho1=%s_model=MINE_hidden=300" % (sample_size, batch_size, rho))
    folder_name_mi_nee = ("pop=%s_batch=%s_Mixed Gaussian_rho1=%s_model=MINEE" % (sample_size, batch_size, rho))

    # Define checkpoint filename and location
    iter_filename = ("cache_iter=%s.pt" % iteration)
    # prefix = os.getcwd()
    # prefix = "C:\\Users\\hphuang\\OneDrive - City University of Hong Kong\\rerun\\pop=1e6"
    prefix = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), "exp")
    chkpt_loc_mine = os.path.join(prefix, folder_name_mine, iter_filename)
    chkpt_loc_mi_nee = os.path.join(prefix, folder_name_mi_nee, iter_filename)

    # Load checkpoint
    checkpoint_mine = torch.load(chkpt_loc_mine, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_mi_nee = torch.load(chkpt_loc_mi_nee, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    #Load ground truth of datasets used in MINE and MI-NEE from log file
    #MINE
    log_filename_mine = "MINE_hidden=300_train.log" 
    log_loc_mine = os.path.join(prefix, folder_name_mine, log_filename_mine)
    ground_truth_mine = find_gt_in_log(log_loc_mine)
    #MI-NEE
    log_filename_mi_nee = "MINEE_train.log"
    log_loc_mi_nee = os.path.join(prefix, folder_name_mi_nee, log_filename_mi_nee)
    ground_truth_mi_nee = find_gt_in_log(log_loc_mi_nee)

    #Load X and Y data used to train MI-NEE
    X_mi_nee = checkpoint_mi_nee['Train_X']
    Y_mi_nee = checkpoint_mi_nee['Train_Y']

    #Load X and Y data used to train MINE
    X_mine = checkpoint_mine['Train_X']
    Y_mine = checkpoint_mine['Train_Y']

    #Set up figure to plot dataset and plot the datasets for MI-NEE and MINE
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_figwidth(15)
    ax1.scatter(X_mi_nee, Y_mi_nee)
    ax1.set_title('Plot of dataset for MI-NEE')
    ax2.scatter(X_mine, Y_mine)
    ax2.set_title('Plot of dataset for MINE')

    ma_rate = ma_rate_box.value #Parameter for moving average for later data smoothing

    # Load MI estimate from MINE
    MI_est_mine = checkpoint_mine['Train_dXY_list']

    #Load joint and marginal estimates from MINE and calculate the MI estimate
    dXY_mi_nee = checkpoint_mi_nee['Train_dXY_list']
    dX_mi_nee = checkpoint_mi_nee['Train_dX_list']
    dY_mi_nee = checkpoint_mi_nee['Train_dY_list']
    MI_est_mi_nee = dXY_mi_nee - dX_mi_nee - dY_mi_nee

    #Apply moving average to smooth out the mutual information estimate
    #MINE
    for i in range(1,MI_est_mine.shape[0]):
        MI_est_mine[i] = (1-ma_rate) * MI_est_mine[i-1] + ma_rate * MI_est_mine[i]
    #MI-NEE
    for i in range(1,dXY_mi_nee.shape[0]):
        MI_est_mi_nee[i] = (1-ma_rate) * MI_est_mi_nee[i-1] + ma_rate * MI_est_mi_nee[i]

    #Set up figure to display MI estimates
    g, (ax1, ax2) = plt.subplots(1, 2)
    g.set_figwidth(15)
    bottom_limit = 0

    #Plot MI-NEE result of MI estimate
    ax1.plot(MI_est_mi_nee)
    top_limit = find_top_limit(ground_truth_mi_nee)
    threshold = find_threshold(0.9, ground_truth_mi_nee)
    ax1.set_ylim(bottom=bottom_limit, top=top_limit)
    ax1.axhline(ground_truth_mi_nee, color="red", linestyle="--", label="ground truth") #Indicate ground truth
    # Assume 90% ground truth is never reached at x = 0, so we can use np.argmax
    # np.argmax will return first x point where the ground truth reaches 90% or above
    MI_90_percent_mi_nee = np.argmax(MI_est_mi_nee >= threshold)
    if (MI_90_percent_mi_nee > 0):
        ax1.axvline(MI_90_percent_mi_nee, color="green", linestyle=":", label="90% reached") #Indicate point where 90% ground truth is reached
    ax1.set_xlabel('number of iterations')
    ax1.set_ylabel('MI estimate')
    ax1.set_title('MI estimate against number of iterations for MI-NEE')
    ax1.legend()

    #Plot MINE result of MI estimate
    ax2.plot(MI_est_mine)
    top_limit = find_top_limit(ground_truth_mine)
    threshold = find_threshold(0.9, ground_truth_mine)
    ax2.set_ylim(bottom=bottom_limit, top=top_limit)
    ax2.axhline(ground_truth_mine, color="red", linestyle="--", label="ground truth")
    # Assume 90% ground truth is never reached at x = 0, so we can use np.argmax
    # np.argmax will return first x point where the ground truth reaches 90% or above
    MI_90_percent_mine = np.argmax(MI_est_mine >= threshold)
    if (MI_90_percent_mine > 0):
        ax2.axvline(MI_90_percent_mine, color="green", linestyle=":", label="90% reached")
    ax2.set_xlabel('number of iterations')
    ax2.set_ylabel('MI estimate')
    ax2.set_title('MI estimate against number of iterations for MINE')
    ax2.legend()

    #Show plots in line
    show_inline_matplotlib_plots()