import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns


def show_object_comparison(prediction, context, name_obj, obj, misvPro, misvCon, output_dir, explanation_type="complete"):
    """ Show object comparison with the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        context: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        obj: The evaluated object.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        output_dir: The output directory to save the plot.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    class_name = ' $\mathbf{' + prediction.class_name + '}$'
    drawing = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
    misvPro = cv2.cvtColor(misvPro, cv2.COLOR_BGR2RGB)
    misvCon = cv2.cvtColor(misvCon, cv2.COLOR_BGR2RGB)

    cArray = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#000000']
    cArrayInv = ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)
    cmInv = ListedColormap(cArrayInv)

    fig = plt.figure(figsize=(6, 4))  # Further reduced width from 8 to 6, and height from 5 to 4

    h0 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    h2 = plt.subplot2grid((4, 5), (0, 2), colspan=3)
    h3 = plt.subplot2grid((4, 5), (1, 2), colspan=3)

    ax0 = plt.subplot2grid((4, 5), (2, 0))
    ax1 = plt.subplot2grid((4, 5), (2, 1), colspan=2)
    ax2 = plt.subplot2grid((4, 5), (2, 3))
    ax3 = plt.subplot2grid((4, 5), (2, 4))
    ax4 = plt.subplot2grid((4, 5), (3, 0), colspan=3)
    ax5 = plt.subplot2grid((4, 5), (3, 3))
    ax6 = plt.subplot2grid((4, 5), (3, 4))

    for ax in [h0, h2, h3, ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')

    h0.imshow(drawing, cmap=cmInv, interpolation='none')
    h2.text(0, 0.5, f'Predicted class: {prediction.class_name}', fontsize=12, style='italic', fontweight='bold', va='center')
    h3.text(0, 0.5, f'Context: {context}', fontsize=12, fontweight='bold', va='bottom')
    ax0.imshow(drawing, cmap=cm, interpolation='none')
    if explanation_type != "unfavorable":
        ax1.text(0, 0.5, f'should be {class_name} because it looks similar to', fontsize=7, va='center', ha='left')
        ax2.imshow(misvPro, cmap=cm, interpolation='none')
        ax3.text(0, 0.5, f', which has been identified as {class_name}.', fontsize=7, va='center', ha='left')
    if explanation_type == "unfavorable":
        ax1.text(0, 0.5, f'should be {class_name}', fontsize=7, va='center', ha='left')
    if explanation_type != "favorable":
        ax4.text(0, 0.5, f'However, the object could not be {class_name} because it also looks like', fontsize=7, va='center', ha='left')
        ax5.imshow(misvCon, cmap=cm, interpolation='none')
        ax6.text(0, 0.5, ', which has not been recognized as such.', fontsize=7, va='center', ha='left')

    plt.tight_layout()
    plt.savefig(f"{output_dir}explaining_prediction-object_{name_obj}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def deskew_image(obj):
    """ Deskew the image.
    
    Parameters:
        obj: The image to deskew.

    Returns:
        img_deskew: The deskewed image.
    """
    height, width = obj.shape[:2]
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    img_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    m = cv2.moments(img_gray)
    if abs(m['mu02']) < 1e-2:
        return img_gray.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * width * skew], [0, 1, 0]])
    img_deskew = cv2.warpAffine(img_gray, M, (width, height), flags=affine_flags)
    return img_deskew


def get_obj_with_influence(obj, fg_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """ Gets the gradient directions with the highest magnitudes of the object.

    Parameters:
        obj: The object to be explained.
        fg_color: The color of the object.
        bg_color: The background color.

    Returns:
        objVectors: The object with influence vectors.
    """
    # Deskew the image
    img_deskew = deskew_image(obj)
    # Calculate gradients
    gx = cv2.Sobel(img_deskew, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_deskew, cv2.CV_32F, 0, 1)

    # Calculate gradient magnitudes and orientations
    magnitude = cv2.magnitude(gx, gy)

    height, width = obj.shape[:2]
    objVectors = np.zeros((height, width, 3), np.uint8)
    objVectors = cv2.rectangle(objVectors, (0, 0), (width, height), bg_color, -1)
    maxX = np.max(np.abs(gx))
    maxY = np.max(np.abs(gy))
    maxXY = max(maxX, maxY)

    # Threshold for significant gradients
    median_magnitude = np.median(magnitude)
    threshold = median_magnitude * 2.5

    for i in range(0, width):
        for j in range(0, height):
            #if (gx[j][i] + gy[j][i]) > 0:
            if magnitude[j, i] > threshold:
                pt1 = (i, j)
                pt2 = (i + int(gx[j, i] / maxXY * 10), j + int(gy[j, i] / maxXY * 10))
                objVectors = cv2.arrowedLine(objVectors, pt1, pt2, fg_color, 1, tipLength=0.3)

    return objVectors


def get_influenceMap(obj, objV, positiveMISV, negativeMISV, bins_n=16, bg_color=(255, 255, 255), explanation_type="complete"):
    """ Get the influence map of the object and its MISVs.

    Parameters:
        obj: The object to be explained.
        objV: The object with influence vectors.
        positiveMISV: The MISV+ of the object.
        negativeMISV: The MISV- of the object.
        bins_n: The number of bins.
        bg_color: The background color.
        explanation_type: The type of influence map (complete, unfavorable, favorable).

    Returns:
        objVectors: The object with influence map.
    """
    # Deskew the image
    img_deskewed = deskew_image(obj)
    height, width = obj.shape[:2]

    gx = cv2.Sobel(img_deskewed, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_deskewed, cv2.CV_32F, 0, 1)

    _, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bins_n * ang / (2 * np.pi))

    proInfluence = objV * positiveMISV
    conInfluence = objV * negativeMISV

    maxProInfluence = np.max(np.abs(proInfluence))
    maxConInfluence = np.max(np.abs(conInfluence))

    proInfluence = proInfluence / max(maxProInfluence, 1)
    conInfluence = conInfluence / max(maxConInfluence, 1)

    objVectors = np.zeros((height, width, 3), np.uint8)
    objVectors = cv2.rectangle(objVectors, (0, 0), (width, height), bg_color, -1)

    maxX = np.max(np.abs(gx))
    maxY = np.max(np.abs(gy))
    maxXY = max(maxX, maxY)

    half_raster_sz = int(min(height, width) / 2)

    magnitude = cv2.magnitude(gx, gy)
    median_magnitude = np.median(magnitude)
    threshold = median_magnitude * 2.5

    for i in range(0, width):
        for j in range(0, height):
            if magnitude[j, i] > threshold:
                offsetInfluence = int((i + j) / half_raster_sz) * bins_n
                idx_angle = bins[j, i]
                influence_index = min(idx_angle + offsetInfluence, len(proInfluence) - 1)

                pt1 = (i, j)
                pt2 = (i + int(gx[j, i] / maxXY * 10), j + int(gy[j, i] / maxXY * 10))

                r = 0
                g = 0
                b = 0

                if (gx[j, i] + gy[j, i]) > 0:
                    if proInfluence[influence_index] + conInfluence[influence_index] != 0:
                        if proInfluence[influence_index] > conInfluence[influence_index]:
                            if explanation_type != "unfavorable":
                                r = 255
                                g = 100 + int(33 * proInfluence[influence_index])
                                b = 64
                        else:
                            if explanation_type != "favorable":
                                r = 0
                                g = 113
                                b = 150 + int(38 * conInfluence[influence_index])
                    else:
                        r = 0
                        g = 0
                        b = 0

                    objVectors = cv2.arrowedLine(objVectors, pt1, pt2, (r, g, b))

    return objVectors


def get_influence(obj, misvPro, misvCon):
    """ Get the influence vectors of the object and its MISVs.

    Parameters:
        obj: The object to be explained.
        misvPro: The MISV+ of the object.
        misvCon: The MISV- of the object.

    Returns:
        objVectors: The object with influence vectors.
        objVectorsPro: The MISV+ with influence vectors.
        objVectorsCon: The MISV- with influence vectors.
    """
    objVectors = get_obj_with_influence(obj)
    objVectorsPro = get_obj_with_influence(misvPro, fg_color=(255, 133, 64), bg_color=(255, 255, 255))
    objVectorsCon = get_obj_with_influence(misvCon, fg_color=(0, 113, 188), bg_color=(255, 255, 255))
    return objVectors, objVectorsPro, objVectorsCon


def show_influence(prediction, influence_map, objVectors, objVectorsPro, objVectorsCon, name_obj, class_pro, class_con,
                   output_dir, explanation_type="complete"):
    """ Show influence map of the object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        influence_map: The influence map of the object.
        objVectors: The evaluated object with influence vectors.
        objVectorsPro: The MISV+ of the evaluated object with influence vectors.
        objVectorsCon: The MISV- of the evaluated object with influence vectors.
        name_obj: The name of the evaluated object.
        class_pro: The class of the MISV+.
        class_con: The class of the MISV-.
        output_dir: The output directory to save the plot.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    cArray = ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1), plt.imshow(objVectors, cmap=cm), plt.axis('off')
    plt.title('obj_{0} ({1})'.format(name_obj, prediction.class_name), style='normal', weight='normal', ha='center',
              size='small')

    if explanation_type != "unfavorable":
        plt.subplot(1, 4, 2), plt.imshow(objVectorsPro, cmap=cm), plt.axis('off')
        plt.title('MISV+ ({0})'.format(class_pro), style='normal', weight='normal', ha='center',
                  size='small')

    if explanation_type == "unfavorable":
        position_con = 2
    else:
        position_con = 3

    if explanation_type != "favorable":
        plt.subplot(1, 4, position_con), plt.imshow(objVectorsCon, cmap=cm), plt.axis('off')
        plt.title('MISV- ({0})'.format(class_con), style='normal', weight='normal', ha='center',
                  size='small')

    if explanation_type == "favorable" or explanation_type == "unfavorable":
        position = 3
    else:
        position = 4
    plt.subplot(1, 4, position), plt.imshow(influence_map), plt.axis('off')
    plt.title('Influence Map', style='normal', weight='normal', ha='center', size='small')
    plt.subplots_adjust(hspace=0.7)
    plt.savefig("{0}explaining_influence-object_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()


def show_influence_map(clf, prediction, obj, objV, misvPro, misvCon, name_obj, class_pro, class_con, output_dir,
                       explanation_type="complete"):
    """ Show influence map with the evaluated object and its MISVs.

    Parameters:
        clf: The classifier used to evaluate the object.
        prediction: The prediction of the evaluated object.
        obj: The object to be explained.
        objV: The object with influence vectors.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        name_obj: The name of the evaluated object.
        class_pro: The class of the MISV+.
        class_con: The class of the MISV-.
        output_dir: The output directory to save the plot.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    SVs = clf.support_vectors_
    positiveMISV = SVs[prediction.eval.mu_hat.misv_idx]
    negativeMISV = SVs[prediction.eval.nu_hat.misv_idx]
    objVectors, objVectorsPro, objVectorsCon = get_influence(obj, misvPro, misvCon)
    if explanation_type == "favorable":
        influence_map = get_influenceMap(obj, objV, positiveMISV, negativeMISV, explanation_type="favorable")
        show_influence(prediction, influence_map, objVectors, objVectorsPro, objVectorsCon, name_obj, class_pro, class_con,
                       output_dir, explanation_type="favorable")
    elif explanation_type == "unfavorable":
        influence_map = get_influenceMap(obj, objV, positiveMISV, negativeMISV, explanation_type="unfavorable")
        show_influence(prediction, influence_map, objVectors, objVectorsPro, objVectorsCon, name_obj, class_pro, class_con,
                       output_dir, explanation_type="unfavorable")
    else:
        SVs = clf.support_vectors_
        positiveMISV = SVs[prediction.eval.mu_hat.misv_idx]
        negativeMISV = SVs[prediction.eval.nu_hat.misv_idx]
        influence_map = get_influenceMap(obj, objV, positiveMISV, negativeMISV)
        show_influence(prediction, influence_map, objVectors, objVectorsPro, objVectorsCon, name_obj, class_pro, class_con,
                       output_dir)


def show_line_chart_comparison(prediction, obj, columns, misvPro, misvCon, context_obj, name_obj, output_dir,
                               classes_map=None, explanation_type="complete"):
    """ Show line chart with the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        obj: The object to be explained.
        columns: The columns of the objects.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        context_obj: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        output_dir: The output directory to save the plot.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Prediction on context {context_obj}: {class_name}', fontsize=14)

    ax.plot(obj, label=f'Evaluated object in context {context_obj}', color='blue')

    if explanation_type != "unfavorable":
        ax.plot(misvPro, label='MISV+', linestyle='dotted', color='orange')

    if explanation_type != "favorable":
        ax.plot(misvCon, label='MISV-', linestyle='dotted', color='green')

    ax.set_title(f'Feature Comparison for object {name_obj}')
    ax.set_ylabel('Levels')
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}explaining_line-chart_{name_obj}.pdf", dpi=300)
    plt.show()


def show_kde_comparison(prediction, obj, misvPro, misvCon, context_obj, name_obj, output_dir, classes_map=None,
                        explanation_type="complete"):
    """ Show KDE plot with the distribution of the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        obj: The object to be explained.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        context_obj: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        output_dir: The output directory to save the plot.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    plt.figure(figsize=(10, 6))
    sns.kdeplot(obj, fill=True, label=f'Object {name_obj}', color='blue')
    if explanation_type != "unfavorable":
        sns.kdeplot(misvPro, fill=True, label='MISV+', color='orange')
    if explanation_type != "favorable":
        sns.kdeplot(misvCon, fill=True, label='MISV-', color='green')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.suptitle(f'Prediction on context {context_obj}: {class_name}')
    plt.title(f'Distribution Comparison between object {name_obj} and MISVs')
    plt.legend()
    plt.grid(True)
    plt.savefig("{0}explaining_kde_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()


def show_heatmap_difference(prediction, obj, columns, misvPro, misvCon, context_obj, name_obj, output_dir,
                            classes_map=None, explanation_type="complete"):
    """ Show heatmap with differences between the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        obj: The object to be explained.
        columns: The columns of the objects.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        context_obj: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        output_dir: The output directory to save the plot.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    index = []
    df_data = []
    if explanation_type != "unfavorable":
        differences = obj - misvPro
        index.append('MISV+')
        df_data.append(differences)
    if explanation_type != "favorable":
        differences2 = obj - misvCon
        index.append('MISV-')
        df_data.append(differences2)

    differences_df = pd.DataFrame(df_data, index=index, columns=columns)
    plt.figure(figsize=(16, 2))
    sns.heatmap(differences_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar=False)
    plt.suptitle(f'Prediction on context {context_obj}: {class_name}', y=1.15)
    plt.title(f'Disparity between object {name_obj} and MISVs')
    plt.savefig("{0}explaining_heatmap_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()


def show_line_chart_difference(prediction, obj, columns, misvPro, misvCon, context_obj, name_obj, output_dir,
                               classes_map=None, explanation_type="complete"):
    """ Show line chart with differences between the evaluated object and its MISVs.
    
    Parameters:
        prediction: The prediction of the evaluated object.
        obj: The object to be explained.
        columns: The columns of the objects.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        context_obj: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        output_dir: The output directory to save the plot.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    differences = obj - misvPro
    differences2 = obj - misvCon
    plt.figure(figsize=(10, 6))
    if explanation_type != "unfavorable":
        plt.plot(differences, marker='o', linestyle='-', color='orange', label='Difference with MISV+')
    if explanation_type != "favorable":
        plt.plot(differences2, marker='x', linestyle='dotted', color='green', label='Difference with MISV-')

    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.ylabel('Difference')
    plt.suptitle(f'Prediction on context {context_obj}: {class_name}')
    plt.title(f'Disparity between object {name_obj} and MISVs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("{0}explaining_line-difference_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()
