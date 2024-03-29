import numpy as np
from scipy.stats import norm
from scipy.stats import dirichlet
import csv, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from GaussianMixDataGenerator.data.randomParameters import NormalMixPNParameters2 as NMixPar
from scipy.stats import dirichlet
from GaussianMixDataGenerator.data.utils import AUCFromDistributions
from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
from ClingenCalibration import calibration
import bisect
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


global outdir

def buildGaussianMixDataGenerator():
    mu_pos = [1.5]
    mu_neg = [-1.5]
    p_pos = [1.0]
    p_neg = [1.0]
    sig_pos = [1.0]
    sig_neg = [1.0]
    alpha = 0.2
    return GMM(mu_pos, sig_pos, p_pos, mu_neg, sig_neg, p_neg, alpha)

def lininterpol(x, x1, x2, y1, y2):
    y = y1 + (x-x1)*( y2-y1 )/(x2-x1)
    return y
    
def getProbabilityBasedOnLinearInterpolation(s, scores, posterior):
    ix= bisect.bisect_left(scores, s)
    ans = None
    if ix == 0:
        ans =  lininterpol(s, scores[0], scores[1], posterior[0], posterior[1])
    elif ix == len(scores):
        ans = lininterpol(s, scores[-1], scores[-2], posterior[-1], posterior[-2])
    else:
        ans = lininterpol(s, scores[ix-1], scores[ix], posterior[ix-1], posterior[ix])

    if ans < 0.0:
        return 0.0
    if ans > 1.0:
        return 1.0
    return ans

def getProbabilitiesBasedOnLinearInterpolation(scores, thresholds, posteriors):
    return [getProbabilityBasedOnLinearInterpolation(e, thresholds, posteriors) for e in scores]


def calibrateModel(scores, label, pudata, alpha):
    thresh, prob = calibration.calibrate(scores, label, pudata, alpha, 100, 0.03)
    thresh.reverse()
    prob = prob.tolist()
    prob.reverse()
    return thresh, prob


def ece(y_test, y_test_pred_class, y_test_pred_prob, M):
    max_p = np.array([max(e,1-e) for e in y_test_pred_prob])
    correct_labels = np.array([1 if y_test[i] == y_test_pred_class[i] else 0 for i in range(len(y_test))])

    print("number correct labels")
    print(np.sum(correct_labels))
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece =  0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.array([i for i in range(len(max_p)) if max_p[i] <= bin_upper.item() and max_p[i] > bin_lower.item()])
        prob_in_bin = in_bin.size/len(max_p)
        if prob_in_bin > 0:
            accuracy_in_bin = correct_labels[in_bin].mean()
            avg_confidence_in_bin = max_p[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece


def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--pnratio_train",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--pnratio_calibrate",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--n_train",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_calibrate",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_test",
        default=None,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--model",
        default="NeuralNetwork",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
    )

    return parser


# Neural Network

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf


def getNNModel():
    NUM_CLASSES = 2
    input_layer = layers.Input((1,))
    
    x = layers.Flatten()(input_layer)
    x = layers.Dense(60, activation="relu")(x)
    
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(input_layer, output_layer)

    return model


def trainModel(model, X_train, y_train):
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(
        loss="binary_crossentropy", optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy"]
    );
    model.fit(X_train, y_train, batch_size=32, epochs=20, shuffle=True);


    
def localCalibration(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob, y_unlabelled_pred_nn_prob, alpha):
    thresh, calibrated_prob = calibrateModel(y_calibrate_pred_nn_prob, y_calibrate.flatten(), y_unlabelled_pred_nn_prob, alpha)
    local_calibrated_prob_test = getProbabilitiesBasedOnLinearInterpolation(y_test_pred_nn_prob,thresh,calibrated_prob)
    return local_calibrated_prob_test

def getPlattCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    logreg = LogisticRegression(class_weight = {0:0.5, 1: 0.5})
    logreg.fit(y_calibrate_pred_nn_prob.reshape(-1,1), y_calibrate.reshape(len(y_calibrate),));
    platt_calibrated_prob_test = [e[1] for e in logreg.predict_proba(y_test_pred_nn_prob.reshape(-1,1))]
    return platt_calibrated_prob_test

def getIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    iso_reg = IsotonicRegression().fit(y_calibrate_pred_nn_prob, y_calibrate.flatten())
    isotonic_calibrated_prob_test = iso_reg.predict(y_test_pred_nn_prob)
    return isotonic_calibrated_prob_test


def plotCalibrationCurve(y_test, local_calibrated_prob_test, platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.get_cmap("Dark2")
    
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), local_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="LocalPosterior", ax=ax_calibration_curve, color=colors(0),)
    calibration_displays["LocalPosterior"] = display
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), platt_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Platt", ax=ax_calibration_curve, color=colors(1),)
    calibration_displays["Platt"] = display
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), isotonic_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Isotonic", ax=ax_calibration_curve, color=colors(2),)
    calibration_displays["Isotonic"] = display
    
    plt.title("pnratio_train: " + str(pnratio_train) + " pnratio_calibrate: " + str(pnratio_calibrate) + " alpha: " + str(alpha))

    title = os.path.join(outdir,"CalibrationCurve_pnratio_" + "pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha) + ".jpg")
    fig.savefig(fname=title)
    plt.close(fig)

def pltPosteriorComparison(true_posterior, local_calibrated_prob_test, platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha):
    plt.title("True Posterior vs Calibrated Posterior")
    plt.scatter(true_posterior,local_calibrated_prob_test, s=1, color='r', label='Local')
    plt.scatter(true_posterior,platt_calibrated_prob_test, s=1, color='g', label='Platt')
    plt.scatter(true_posterior,isotonic_calibrated_prob_test, s=1, color='orange', label='Isotonic')
    plt.plot(true_posterior,true_posterior, color= 'black', label='True')
    plt.legend(loc='best')
    fname = os.path.join(outdir, "CalibratedPosterior_pnratio_train" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha)  + ".jpg")
    plt.savefig(fname)
    plt.close()


    
def toynn(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu):
    
    model = getNNModel()
    trainModel(model, X_train, y_train)

    y_calibrate_pred_nn_prob = model(X_calibrate, training=False).numpy().flatten()
    y_test_pred_nn_prob = model(X_test, training=False).numpy().flatten()
    y_unlabelled_pred_nn_prob = model(xu, training=False).numpy().flatten()

    return y_calibrate_pred_nn_prob, y_test_pred_nn_prob, y_unlabelled_pred_nn_prob
    #local_calibrated_prob_test = y_test_pred_nn_prob



def toyrf(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu):

    model = RandomForestClassifier(n_estimators=100,)
    model.fit(X_train, y_train);

    y_calibrate_pred_rf_prob = model.predict_proba(X_calibrate)[:, 1].flatten()
    y_test_pred_rf_prob = model.predict_proba(X_test)[:, 1].flatten()
    y_unlabelled_pred_rf_prob = model.predict_proba(xu)[:, 1].flatten()

    return y_calibrate_pred_rf_prob, y_test_pred_rf_prob, y_unlabelled_pred_rf_prob



def main():
    
    global outdir
    
    parser = getParser()
    args = parser.parse_args()
    outdir = args.outdir
    alpha = args.alpha
    pnratio_train = args.pnratio_train
    pnratio_calibrate = args.pnratio_calibrate
    
    gmm = buildGaussianMixDataGenerator()
    #x,y = gmm.pn_data(20000, pnratio)[0:2]

    n_train = 20000
    n_calibrate = 5000
    n_test = 5000

    outdir = os.path.join(outdir, args.model)

    if args.n_train is not None:
        n_train = args.n_train
        outdir = os.path.join(outdir,"n_train_"+ str(n_train))
    if args.n_calibrate is not None:
        n_calibrate = args.n_calibrate
        outdir = os.path.join(outdir,"n_calibrate_" + str(n_calibrate))
    if args.n_test is not None:
        n_test = args.n_test
        outdir = os.path.join(outdir, "n_test_" + str(n_test))

    os.makedirs(outdir, exist_ok=True)
    
    X_train, y_train = gmm.pn_data(n_train, pnratio_train)[0:2]
    
    list1, list2 = (list(t) for t in zip(*sorted(zip(X_train, y_train))))
    print("auc:", auc(list1, list2))
    
    X_calibrate, y_calibrate = gmm.pn_data(n_calibrate, pnratio_calibrate)[0:2]
    X_test, y_test = gmm.pn_data(n_test, alpha)[0:2]

    xu, yu = gmm.pn_data(40000, alpha)[0:2]

    #X_train, X_test_and_calibrate, y_train, y_test_and_calibrate = train_test_split(x, y, test_size=0.5, random_state=16)
    #X_test, X_calibrate, y_test, y_calibrate = train_test_split(X_test_and_calibrate, y_test_and_calibrate, test_size=0.5, random_state=16)
    true_posterior = gmm.pn_posterior(X_test, alpha)
    true_posterior_pnratio = gmm.pn_posterior(X_test, alpha)

    y_calibrate_pred_prob = None
    y_test_pred_prob = None
    y_unlabelled_pred_prob = None
    
    if args.model == 'NeuralNetwork':
        y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob = toynn(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu)

    if args.model == "RandomForest":
        y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob = toyrf(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu)

    #local_calibrated_prob_test = y_test_pred_prob
    local_calibrated_prob_test = localCalibration(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob, alpha)
    platt_calibrated_prob_test = getPlattCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob);
    isotonic_calibrated_prob_test = getIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob)

    plotCalibrationCurve(y_test, local_calibrated_prob_test, platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha)
    pltPosteriorComparison(true_posterior, local_calibrated_prob_test, platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha)

    print(y_test_pred_prob)
    
    y_test_pred_class_local = [1 if e > 0.5 else 0 for e in local_calibrated_prob_test]
    
    dometrics(y_test, y_test_pred_class_local, local_calibrated_prob_test, 10)
    plt.scatter(X_test, y_calibrate_pred_prob)
    plt.savefig("checkthis.jpg")
    plt.close()

    '''
    samples = np.array([[0.78, 0.22],
                    [0.36, 0.64],
                    [0.08, 0.92],
                    [0.58, 0.42],
                    [0.49, 0.51],
                    [0.85, 0.15],
                    [0.30, 0.70],
                    [0.63, 0.37],
                    [0.17, 0.83]])
    '''
    a = [0,1,1,0,1,0,1,0,1]
    b = [0.22,0.64,0.92,0.42,0.51,0.15,0.7,0.37,0.83]
    true_labels = np.array([0,1,0,0,0,0,1,1,1])
    dometrics(true_labels, a, b , 5 )
    
    
def dometrics(y_test, y_test_pred_class, y_test_pred_prob, M):
    ecev = ece(y_test, y_test_pred_class, y_test_pred_prob, M)
    print("ecev")
    print(ecev)
    
main()
