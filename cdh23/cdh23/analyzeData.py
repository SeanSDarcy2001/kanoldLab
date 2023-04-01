from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import animation 
from matplotlib.animation import FFMpegFileWriter
import cdh23.animator
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from scipy import stats
import itertools

class analyzeTheData:

    #data is 200 trials for one mouse
    def __init__(self, data, stim, mouse, out_dir) :
        self.mouse = mouse
        self.directory = out_dir
        #set output dir accordingly
        self.output_dir = Path("/Volumes/Data2/Travis/Cdh23/Sean/outputs", self.directory, self.mouse)
        isExist = os.path.exists(self.output_dir)
        if not isExist:
        # Create a new directory because it does not exist
            os.makedirs(self.output_dir)
        self.data = data
        self.frames_pre_stim = 14 #24
        self.frames_post_stim = 45
        self.start_stim = 15
        self.end_stim = 23
        self.trial_type   = stim[0][:, 0] #only 0 for first session, #stimHistories[mou] is 200x3, len 200
        self.trial_types  = ["4 kHz", "8 kHz", "16 kHz", "32 kHz", "64 kHz"]
        self.attenuations = ["0 dB", "20 dB", "40 dB", "60"]
        self.trials       = self.data #len 200
        self.trial_size   = self.trials[0].shape[1] #was trials[0]
        self.Nneurons     = self.trials[0].shape[0]
        # list of arrays containing the indices of trials of each type (t_type_ind[0] contains the
        # indices of trials of type trial_types[0])
        #t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:] for t_type in trial_types]
        self.t_type_ind = [range(40),range(40,80), range(80, 120), range(120, 160), range(160,200)]
        self.a_type_ind = [range(10),range(10,20), range(20, 30), range(30, 40)]


        self.shade_alpha      = 0.2
        self.lines_alpha      = 0.8
        self.pal = sns.color_palette('husl', 9)
        #%config InlineBackend.figure_format = 'svg'


    def z_score(self, X):
        # X: ndarray, shape (n_features, n_samples)
        ss = StandardScaler(with_mean=True, with_std=True)
        Xz = ss.fit_transform(X.T).T
        return Xz
    

    def add_stim_to_plot(self, ax):
        ax.axvspan(self.start_stim, self.end_stim, alpha=self.shade_alpha,
               color='gray')
        ax.axvline(self.start_stim, alpha=self.lines_alpha, color='gray', ls='--')
        ax.axvline(self.end_stim, alpha=self.lines_alpha, color='gray', ls='--')
    
    def add_orientation_legend(self, ax):
        custom_lines = [Line2D([0], [0], color=self.pal[k], lw=4) for
                    k in range(len(self.trial_types))]
        labels = ['{}$^\circ$'.format(t) for t in self.trial_types]
        ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0,0,0.9,1])
    
    #Trial-Response PCA: Reduce each trial to a single sample of shape N×1, 
    # where N is the number of neurons recorded. These samples are stacked to produce a 
    # matrix of shape N×K where K is the number of trials. 
    def trial_response_pca(self) :
        #Xr = np.vstack([t[:, self.frames_pre_stim:self.frames_post_stim].mean(axis=1) for t in self.trials]).T
        # or take the max
        # or take the max
        Xr = np.vstack([t[:, self.frames_pre_stim:self.frames_pre_stim+20].max(axis=1) for t in self.trials]).T
        #Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].max(axis=1) for t in trials]).T
        # or the baseline-corrected mean
        #Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].mean(axis=1) - t[:, 0:frames_pre_stim].mean(axis=1) for t in trials]).T

        Xr_sc = self.z_score(Xr)
        pca = PCA(n_components=15)
        Xp = pca.fit_transform(Xr_sc.T).T
        projections = [(0, 1), (1, 2), (0, 2)] #projections = [(2, 0), (2, 1), (2, 3), (2, 4), (2,5)]
        fig, axes = plt.subplots(1, len(projections), figsize=[9, 3], sharey='row', sharex='row')

        counter = 0
        xhats = np.zeros((3, 200, 2))
        for ax, proj in zip(axes, projections):
            print("PCs:", proj)
            for t, t_type in enumerate(self.trial_types):
                print("t_type:", t_type)
                x = Xp[proj[0], self.t_type_ind[t]]
                y = Xp[proj[1], self.t_type_ind[t]]
                
                xhats[counter, self.t_type_ind[t], 0] = x
                xhats[counter, self.t_type_ind[t], 1] = y

                ax.scatter(x, y, c=self.pal[t], s=25, alpha=0.8)
                ax.set_xlabel('PC {}'.format(proj[0]+1))
                ax.set_ylabel('PC {}'.format(proj[1]+1))
            counter = counter + 1 #for dataset

        sns.despine(fig=fig, top=True, right=True)
        self.add_orientation_legend(axes[len(projections)-1])
        plt.savefig(Path(self.output_dir, "trial_responseWmax"))
        return xhats #returns data for classification


    #Trial Averaged PCA: First average all trials of each type together, 
    # concatenate the trial averages, and then apply PCA to the resulting matrix.
    def trial_average_pca(self):
        trial_averages = []

        for ind in self.t_type_ind:
            ind = list(ind)
            i = 0
            for indeces in ind :
                if i == 0 :
                    sum = self.trials[indeces]
                    i = i + 1
                else :
                    sum = sum + self.trials[indeces]
            mean = sum / len(ind)
            trial_averages.append(mean)
        Xa = np.hstack(trial_averages)

        n_components = 15
        Xa = self.z_score(Xa) #Xav_sc = center(Xav)
        pca = PCA(n_components=n_components)
        Xa_p = pca.fit_transform(Xa.T).T

        numComponents = 15 #set how many to plot
        fig, axes = plt.subplots(1, numComponents, figsize=[50, 2.8], sharey='row')
        for comp in range(numComponents):
            ax = axes[comp]
            for kk, type in enumerate(self.trial_types):
                x = Xa_p[comp, kk * self.trial_size :(kk+1) * self.trial_size]
                x = gaussian_filter1d(x, sigma=1)
                ax.plot(np.linspace(0,len(x), len(x)), x, c=self.pal[kk])
            self.add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp+1))
        self.add_orientation_legend(axes[numComponents-1])
        axes[1].set_xlabel('Frame')
        sns.despine(fig=fig, right=True, top=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(Path(self.output_dir, "trial_average"))

    def style_3d_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')


    def threeDtrajectories(self) :
        trial_averages = []
        for ind in self.t_type_ind:
            ind = list(ind)
            i = 0
            for indeces in ind :
                if i == 0 :
                    sum = self.trials[indeces]
                    i = i + 1
                else :
                    sum = sum + self.trials[indeces]
            mean = sum / len(ind)
            trial_averages.append(mean)

        Xa = np.hstack(trial_averages)

        # standardize and apply PCA
        Xa = self.z_score(Xa) 
        pca = PCA(n_components=3)
        Xa_p = pca.fit_transform(Xa.T).T

        # pick the components corresponding to the x, y, and z axes
        component_x = 0
        component_y = 1
        component_z = 2

        # create a boolean mask so we can plot activity during stimulus as 
        # solid line, and pre and post stimulus as a dashed line
        stim_mask = ~np.logical_and(np.arange(self.trial_size) >= self.frames_pre_stim,
               np.arange(self.trial_size) < (self.trial_size-self.frames_post_stim))
        
        sigma = 1.5 # smoothing amount

        # set up a figure with two 3d subplots, so we can have two different views
        fig = plt.figure(figsize=[10, 4])
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        axs = [ax1, ax2]

        for ax in axs:
            for t, t_type in enumerate(self.trial_types):

                # for every trial type, select the part of the component
                # which corresponds to that trial type:
                x = Xa_p[component_x, t * self.trial_size :(t+1) * self.trial_size]
                y = Xa_p[component_y, t * self.trial_size :(t+1) * self.trial_size]
                z = Xa_p[component_z, t * self.trial_size :(t+1) * self.trial_size]
        
                # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)
                z = gaussian_filter1d(z, sigma=sigma)

                # use the mask to plot stimulus and pre/post stimulus separately
                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan

                ax.plot(x, y, z_stim, c = self.pal[t])
                ax.plot(x, y, z_prepost, c=self.pal[t], ls=':')

                # plot dots at initial point
                ax.scatter(x[0], y[0], z[0], c=self.pal[t], s=14)
        
                # make the axes a bit cleaner
                self.style_3d_ax(ax)
        
                # specify the orientation of the 3d plot        
        ax1.view_init(elev=22, azim=30)
        ax2.view_init(elev=22, azim=87)
        plt.tight_layout()
        plt.savefig(Path(self.output_dir, "3d_traj"))
        self.animated3D(Xa_p, sigma)


    def animated3D(self, Xa_p, sigma) :       
        # apply some smoothing to the trajectories
        for c in range(Xa_p.shape[0]):
            Xa_p[c, :] =  gaussian_filter1d(Xa_p[c, :], sigma=sigma)
        # create the figure
        fig = plt.figure(figsize=[9, 9]); plt.close()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        animator = cdh23.animator.animater(Xa_p, ax, self.trial_size, self.trial_types, self.frames_pre_stim, self.frames_post_stim,self.pal)
        anim = animation.FuncAnimation(fig, animator.animate,
                               frames=68, interval=50, 
                               blit=True)
        writervideo = animation.FFMpegFileWriter(fps=15) 
        anim.save(Path(self.output_dir, "traj_video.mp4"), writer=writervideo)

    def twoD_vid(self) :
        # fit PCA on trial averages
        n_components = 15
        trial_averages = []
        for ind in self.t_type_ind:
            ind = list(ind)
            i = 0
            for indeces in ind :
                if i == 0 :
                    sum = self.trials[indeces]
                    i = i + 1
                else :
                    sum = sum + self.trials[indeces]
            mean = sum / len(ind)
            trial_averages.append(mean)
        Xav = np.hstack(trial_averages)
        ss = StandardScaler(with_mean=True, with_std=True)
        Xav_sc = ss.fit_transform(Xav.T).T
        pca = PCA(n_components) 
        pca.fit(Xav_sc.T) # only call the fit method

        projected_trials = []
        for trial in self.trials:
            # scale every trial using the same scaling applied to the averages 
            trial = ss.transform(trial.T).T
            # project every trial using the pca fit on averages
            proj_trial = pca.transform(trial.T).T
            projected_trials.append(proj_trial)

        gt = {comp: {t_type: [] for t_type in self.trial_types}
            for comp in range(n_components)}

        for comp in range(n_components):
            for i, t_type in enumerate(self.trial_types):
                t = projected_trials[i][comp, :]
                gt[comp][t_type].append(t)

        # smooth the single projected trials 
        for i in range(len(projected_trials)):
            for c in range(projected_trials[0].shape[0]):
                projected_trials[i][c, :] = gaussian_filter1d(projected_trials[i][c, :], sigma=1)

        # for every time point (imaging frame) get the position in PCA space of every trial
        pca_frame = []
        for t in range(self.trial_size):
            # projected data for all trials at time t 
            Xp = np.hstack([tr[:, None, t] for tr in projected_trials]).T
            pca_frame.append(Xp)

        subspace = (1,0) # pick the subspace given by the 1st and 2nd components

    
        # set up the figure
        fig, ax = plt.subplots(1, 1, figsize=[6, 6]); plt.close()
        ax.set_xlim(( -100, 100))
        ax.set_ylim((-100, 100))
        ax.set_xlabel('PC 1')
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-100, 0, 100])
        ax.set_ylabel('PC 2')
        sns.despine(fig=fig, top=True, right=True)

        # generate empty scatter plot to be filled by data at every time point
        scatters = []

        for t, t_type in enumerate(self.trial_types):
            scatter, = ax.plot([], [], 'o', lw=2, color=self.pal[t])
            scatters.append(scatter)
        # red dot to indicate when stimulus is being presented
        stimdot, = ax.plot([], [], 'o', c='r', markersize=35, alpha=0.5)
        text     = ax.text(55, 60, 'Stimulus OFF', fontdict={'fontsize':14})
        # annotate with stimulus and time information
        #text     = ax.text(6.3, 9, 'Stimulus OFF \nt = {:.2f}'.format(time[0]), fontdict={'fontsize':14})

        def animate(i):
            for t, t_type in enumerate(self.trial_types):
                # find the x and y position of all trials of a given type
                x = pca_frame[i][self.t_type_ind[t], subspace[0]]
                y = pca_frame[i][self.t_type_ind[t], subspace[1]]
                # update the scatter
                scatters[t].set_data(x, y)
        
            # update stimulus and time annotation
            if (i >= self.frames_pre_stim) and (i < (self.trial_size-self.frames_post_stim)):
                stimdot.set_data(80, 80)
                text.set_text('Stimulus ON')
            else:
                stimdot.set_data([], [])
                text.set_text('Stimulus OFF')
            return (scatter,)
        
        # generate the animation
        anim = animation.FuncAnimation(fig, animate, 
                               frames=len(pca_frame), interval=80, 
                               blit=False)

        writervideo = animation.FFMpegFileWriter(fps=15) 
        anim.save(Path(self.output_dir, "2dPC_video.mp4"), writer=writervideo)

        # set up a dictionary to color each line
        col = {self.trial_types[i] : self.pal[i] for i in range(len(self.trial_types))}


        fig = plt.figure(figsize=[9, 9]); plt.close()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        def animate2(i):

            component_x = 0
            component_y = 1
            component_z = 2
    
            ax.clear()
            self.style_3d_ax(ax)
            ax.view_init(elev=22, azim=30)
            for t, (trial, t_type) in enumerate(zip(projected_trials[0:200], self.trial_type[0:200])):
        
                x = trial[component_x, :][0:i]
                y = trial[component_y, :][0:i]
                z = trial[component_z, :][0:i]
        
                stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= self.frames_pre_stim,
                     np.arange(z.shape[0]) < (self.trial_size-self.frames_pre_stim))

                z_stim = z.copy()
                z_stim[stim_mask] = np.nan
                z_prepost = z.copy()
                z_prepost[~stim_mask] = np.nan
        
                ax.plot(x, y, z_stim, c = col[self.trial_types[t_type-1]])
                ax.plot(x, y, z_prepost, c=col[self.trial_types[t_type-1]], ls=':')

            ax.set_xlim(( -100, 100))
            ax.set_ylim((-100, 100))
            ax.set_zlim((-100, 100))
            ax.view_init(elev=22, azim=30)

            return []

        anim = animation.FuncAnimation(fig, animate2, frames=len(pca_frame), interval=50,blit=True)
        anim.save(Path(self.output_dir, "avg_concat_video.mp4"), writer=writervideo)

    def getFeature_Neurons(self, pc) :
            trial_averages = []
            for ind in self.t_type_ind:
                ind = list(ind)
                i = 0
                for indeces in ind :
                    if i == 0 :
                        sum = self.trials[indeces]
                        i = i + 1
                    else :
                        sum = sum + self.trials[indeces]
                mean = sum / len(ind)
                trial_averages.append(mean)
            Xa = np.hstack(trial_averages)

            n_components = 15
            Xa = self.z_score(Xa) #Xav_sc = center(Xav)
            pca = PCA(n_components=n_components)
            Xa_p = pca.fit_transform(Xa.T).T
            # find the indices of the three largest elements of the eigenvector of interest
            component_of_interest = pc #set ev of interest
            units = np.abs(pca.components_[component_of_interest-1, :].argsort())[::-1][0:10]
            f, axes = plt.subplots(1, 10, figsize=[50, 2.8], sharey=False,
                       sharex=True)
            for ax, unit in zip(axes, units):
                ax.set_title('Neuron {}'.format(unit))
                for t, ind in enumerate(self.t_type_ind):
                    ind = list(ind)
                    x = np.zeros((self.trials[0].shape[1]))
                    k = 0
                    for indeces in ind :
                        x = np.add(x, self.trials[indeces][unit, :])
                        k = k + 1
                    x = x/k
                    ax.errorbar(np.linspace(0,len(x), len(x)), x, yerr=np.std(x), c=self.pal[t], elinewidth=.5)
            for ax in axes:
                self.add_stim_to_plot(ax)
            axes[1].set_xlabel('Frame')
            sns.despine(fig=f, right=True, top=True)
            self.add_orientation_legend(axes[2])

    #test percent is test subset, pcs is set of pcs to do classification, data is from trialresponse
    def classify(self, testPercent, pcs, data) :
        
        trainingData = data
        pcscape = pcs
        trials = 200
        test_sz=testPercent
        testTrials = test_sz*trials

        print("Trials:", trials, "|| Test %:", test_sz, "|| Test Trials:", testTrials)

        y = np.zeros(trials)
        for i in range(trials) :
            if i < 40 :
                y[i] = 0
            elif i < 80 :
                y[i] = 1
            elif i < 120 :
                y[i] = 2
            elif i < 160 :
                y[i] = 3
            elif i < 200 :
                y[i] = 4

        X_train, X_test, y_train, y_test = train_test_split(trainingData[pcscape], y, test_size = test_sz, random_state=2)
        #classifier = RandomForestClassifier(max_depth=2, random_state=1)
        classifier = svm.SVC(decision_function_shape = 'ovo')

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if X_test[:, 0].min() < X_train[:, 0].min() :
            zerMin = X_test[:, 0].min()
        else :
            zerMin = X_train[:, 0].min()

        if X_test[:, 1].min() < X_train[:, 1].min() :
            oneMin = X_test[:, 1].min()
        else :
            oneMin = X_train[:, 1].min()

        if X_test[:, 0].max() > X_train[:, 0].max() :
            zerMax = X_test[:, 0].max()
        else :
            zerMax = X_train[:, 0].max()

        if X_test[:, 1].max() > X_train[:, 1].max() :
            oneMax = X_test[:, 1].max()
        else :
            oneMax = X_train[:, 1].max()
            
        feature_1, feature_2 = np.meshgrid(
            np.linspace(zerMin, zerMax),
            np.linspace(oneMin, oneMax))
        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
        deb = np.reshape(classifier.predict(grid), feature_1.shape)
        display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=deb)
        display.plot( alpha=0.6)
        display.ax_.scatter(X_test[:, 0], X_test[:, 1], c = y_test,s=20)
        plt.title("SVM Decision Boundaries and Test Data")
        nam = "SVM" + str(pcscape)
        plt.savefig(Path(self.output_dir, nam))

        acc_labels = np.zeros(len(y_pred))
        for i in range(len(y_pred)) :
            if y_pred[i] == y_test[i] :
                acc_labels[i] = 1

        cdict = {0.0: 'red', 1.0: 'green'}

        fig, ax = plt.subplots()
        for g in np.unique(acc_labels):
            ix = np.where(acc_labels == g)
            ax.scatter(X_test[ix, 0], X_test[ix, 1], c = cdict[g], s = 100)

        ax.legend(['Incorrect', "Correct"])
        plt.title("Classification Accuracy: " + str(accuracy_score(y_test, y_pred)))
        
        nam = "accPlot" + str(pcscape)
        plt.savefig(Path(self.output_dir, nam))

        plt.figure()
        plt.xlabel("Predicted Tone")
        plt.ylabel("True Tone")
        plt.imshow(cm)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                        plt.text(j, i, cm[i, j],
                        horizontalalignment="center")
        plt.colorbar()
        plt.yticks([0, 1, 2, 3, 4], ['4 kHz', '8 kHz', '16 kHz', '32 kHz', '64 kHz'])
        plt.xticks([0, 1, 2, 3, 4], ['4 kHz', '8 kHz', '16 kHz', '32 kHz', '64 kHz'])
        pval = stats.binom_test(testTrials * accuracy_score(y_test, y_pred), n=testTrials, p=.2, alternative='greater')
        print('Accuracy:', str(accuracy_score(y_test, y_pred)))
        print("P-Value:", pval)
        plt.title("Confusion Matrix, p =" + str(pval))
        nam = "confusion_matrix" + str(pcscape)
        plt.savefig(Path(self.output_dir, nam))

