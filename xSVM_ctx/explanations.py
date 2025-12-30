import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def show_object_comparison(prediction, context, name_obj, obj, misvPro, misvCon, output_dir, classes_map=None,
                           explanation_type="complete"):
    """ Show object comparison with the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        context: The context of the evaluated object.
        name_obj: The name of the evaluated object.
        obj: The evaluated object.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        output_dir: The output directory to save the plot.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """

    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    class_name = ' $\mathbf{' + class_name + '}$'
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
            # if (gx[j][i] + gy[j][i]) > 0:
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


def show_word_comparison(prediction, context, name_obj, obj, misvPro, misvCon, vectorizer, output_dir, method='pca',
                         perplexity=30, random_state=42, classes_map=None, explanation_type="complete"):
    """ Show the comparison of word embeddings between the evaluated object and its MISVs.

    Parameters:
        prediction: The prediction of the evaluated object.
        context: The context of evaluated object.
        name_obj: The name of the evaluated object.
        obj: The object to be explained.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        vectorizer: The vectorizer used on the sentences.
        output_dir: The output directory to save the plot.
        method: Dimensionality reduction method (pca, tsne).
        perplexity: Perplexity parameter for TSNE.
        random_state: Random state for the dimensionality reduction method.
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    from adjustText import adjust_text

    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    if explanation_type == 'favorable':
        texts = [obj, misvPro]
        text_names = [name_obj, 'MISV+']
    elif explanation_type == 'unfavorable':
        texts = [obj, misvCon]
        text_names = [name_obj, 'MISV-']
    else:  # complete
        texts = [obj, misvPro, misvCon]
        text_names = [name_obj, 'MISV+', 'MISV-']

    tfidf_matrix = vectorizer.transform(texts)
    tfidf_dense = tfidf_matrix.toarray()

    tfidf_matrix = vectorizer.transform(texts)
    tfidf_dense = tfidf_matrix.toarray()

    words = vectorizer.get_feature_names_out()

    word_vectors = tfidf_dense.T

    word_counts = (word_vectors > 0).sum(axis=1)
    word_mask = word_counts >= 1
    word_vectors_filtered = word_vectors[word_mask]
    words_filtered = words[word_mask]

    if len(words_filtered) == 0:
        raise ValueError("No words found")

    word_vectors_normalized = normalize(word_vectors_filtered, axis=1)

    word_to_sentences = {}
    for word_idx, word in enumerate(words_filtered):
        sentence_indices = set(np.where(word_vectors_filtered[word_idx] > 0)[0].tolist())
        word_to_sentences[word] = sentence_indices

    if explanation_type == 'favorable':
        color_map = {
            frozenset([0]): '#5E6969',  # Only Obj (Gray)
            frozenset([1]): '#F39C12',  # Only MISV+ (Orange)
            frozenset([0, 1]): '#2ECC71'  # Shared (Green)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV+ only', '#F39C12'),
            (f'Shared: Object {name_obj} & MISV+', '#2ECC71')
        ]
    elif explanation_type == 'unfavorable':
        color_map = {
            frozenset([0]): '#5E6969',  # Only Obj (Gray)
            frozenset([1]): '#3498DB',  # Only MISV- (Blue)
            frozenset([0, 1]): '#9B59B6'  # Shared (Purple)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV- only', '#3498DB'),
            (f'Shared: Object {name_obj} & MISV-', '#9B59B6')
        ]
    else:  # complete
        color_map = {
            frozenset([0]): '#5E6969',  # Only Obj (Gray)
            frozenset([1]): '#F39C12',  # Only MISV+ (Orange)
            frozenset([2]): '#3498DB',  # Only MISV- (Blue)
            frozenset([0, 1]): '#2ECC71',  # Obj & MISV+ (Green)
            frozenset([0, 2]): '#9B59B6',  # Obj & MISV- (Purple)
            frozenset([1, 2]): '#E74C3C',  # MISV+ & MISV- (Red)
            frozenset([0, 1, 2]): '#E67E22'  # All three (Dark Orange)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV+ only', '#F39C12'),
            ('MISV- only', '#3498DB'),
            (f'Shared: Object {name_obj} & MISV+', '#2ECC71'),
            (f'Shared: Object {name_obj} & MISV-', '#9B59B6'),
            ('Shared: All Three', '#E67E22')
        ]

    word_colors = []
    word_categories = []
    words_to_plot = []
    word_vectors_to_plot = []

    for idx, word in enumerate(words_filtered):
        sentence_set = frozenset(word_to_sentences[word])

        if explanation_type == 'complete' and sentence_set == frozenset([1, 2]):
            continue

        color = color_map.get(sentence_set, '#5E6969')
        word_colors.append(color)
        word_categories.append(sentence_set)
        words_to_plot.append(word)
        word_vectors_to_plot.append(word_vectors_normalized[idx])

    words_filtered = np.array(words_to_plot)
    word_vectors_normalized = np.array(word_vectors_to_plot)

    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        word_vectors_2d = reducer.fit_transform(word_vectors_normalized)
        explained_var = reducer.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    else:  # tsne
        perplexity_adjusted = min(perplexity, len(words_filtered) - 1)
        reducer = TSNE(n_components=2, random_state=random_state,
                       perplexity=perplexity_adjusted, init='pca')
        word_vectors_2d = reducer.fit_transform(word_vectors_normalized)

    plt.figure(figsize=(16, 12))

    texts_to_adjust = []
    for i, word in enumerate(words_filtered):
        color = word_colors[i]
        category = word_categories[i]

        is_shared = len(category) > 1
        size = 100 if is_shared else 60
        alpha = 0.7 if is_shared else 0.5
        edgecolor = 'black' if is_shared else 'gray'
        linewidth = 1.0 if is_shared else 0.5

        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1],
                    color=color, s=size, alpha=alpha,
                    edgecolors=edgecolor, linewidth=linewidth, zorder=2)

        fontweight = 'bold' if is_shared else 'normal'
        fontsize = 11 if is_shared else 9

        text = plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word,
                        color=color, ha='center', fontsize=fontsize,
                        fontweight=fontweight, zorder=3)
        texts_to_adjust.append(text)

    gca = plt.gca()
    adjust_text(texts_to_adjust,
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.5),
                ax=gca)

    method_name = method.upper()

    plt.suptitle(f'Prediction on context {context}: {class_name}')
    plt.title(f'Word Embedding Comparison ({method_name}) between object {name_obj} and MISVs')

    plt.xlabel(f"Dimension 1", fontsize=12)
    plt.ylabel(f"Dimension 2", fontsize=12)
    plt.grid(alpha=0.3)

    legend_elements = []
    for label, color in legend_labels:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10,
                       label=label, markeredgewidth=1.5,
                       markeredgecolor='black')
        )

    plt.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("{0}word_comparison_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()


def show_hierarchical_clustering(prediction, context, name_obj, obj, misvPro, misvCon, vectorizer, output_dir,
                                 method='ward', metric='euclidean', classes_map=None, explanation_type="complete"):
    """
    Show the hierarchical clustering of words between the evaluated object and its MISVs.
    Parameters:
        prediction: The prediction of the evaluated object.
        context: The context of evaluated object.
        name_obj: The name of the evaluated object.
        obj: The object to be explained.
        misvPro: The MISV+ of the evaluated object.
        misvCon: The MISV- of the evaluated object.
        vectorizer: The vectorizer used on the sentences.
        output_dir: The output directory to save the plot.
        method: Dimensionality reduction method (pca, tsne).
        classes_map: The classes mapping to show the class name.
        explanation_type: The type of explanation to show (complete, unfavorable, favorable).
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    if classes_map:
        class_name = classes_map[prediction.class_name]
    else:
        class_name = prediction.class_name

    if explanation_type == 'favorable':
        texts = [obj, misvPro]
        text_names = [name_obj, 'MISV+']
    elif explanation_type == 'unfavorable':
        texts = [obj, misvCon]
        text_names = [name_obj, 'MISV-']
    else:  # complete
        texts = [obj, misvPro, misvCon]
        text_names = [name_obj, 'MISV+', 'MISV-']

    tfidf_matrix = vectorizer.transform(texts)
    tfidf_dense = tfidf_matrix.toarray()

    tfidf_matrix = vectorizer.transform(texts)
    tfidf_dense = tfidf_matrix.toarray()

    words = vectorizer.get_feature_names_out()

    word_vectors = tfidf_dense.T

    word_counts = (word_vectors > 0).sum(axis=1)
    word_mask = word_counts >= 1
    word_vectors_filtered = word_vectors[word_mask]
    words_filtered = words[word_mask]

    if len(words_filtered) == 0:
        raise ValueError("No words found")

    word_to_sentences = {}
    for word_idx, word in enumerate(words_filtered):
        sentence_indices = set(np.where(word_vectors_filtered[word_idx] > 0)[0].tolist())
        word_to_sentences[word] = sentence_indices

    if explanation_type == 'favorable':
        color_map = {
            frozenset([0]): '#5E6969',  # Only Evaluated (Gray)
            frozenset([1]): '#F39C12',  # Only Favorable (Orange)
            frozenset([0, 1]): '#2ECC71'  # Shared (Green)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV+ only', '#F39C12'),
            (f'Shared: Object {name_obj} & MISV+', '#2ECC71')
        ]
        shared_colors = ['#2ECC71']
    elif explanation_type == 'unfavorable':
        color_map = {
            frozenset([0]): '#5E6969',  # Only Evaluated (Gray)
            frozenset([1]): '#3498DB',  # Only Unfavorable (Blue)
            frozenset([0, 1]): '#9B59B6'  # Shared (Purple)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV- only', '#3498DB'),
            (f'Shared: Object {name_obj} & MISV-', '#9B59B6')
        ]
        shared_colors = ['#9B59B6']
    else:  # complete
        color_map = {
            frozenset([0]): '#5E6969',  # Only Evaluated (Gray)
            frozenset([1]): '#F39C12',  # Only Favorable (Orange)
            frozenset([2]): '#3498DB',  # Only Unfavorable (Blue)
            frozenset([0, 1]): '#2ECC71',  # Evaluated & Favorable (Green)
            frozenset([0, 2]): '#9B59B6',  # Evaluated & Unfavorable (Purple)
            frozenset([1, 2]): '#E74C3C',  # Favorable & Unfavorable (Red)
            frozenset([0, 1, 2]): '#E67E22'  # All three (Dark Orange)
        }
        legend_labels = [
            (f'Object {name_obj} only', '#5E6969'),
            ('MISV+ only', '#F39C12'),
            ('MISV- only', '#3498DB'),
            (f'Shared: Object {name_obj} & MISV+', '#2ECC71'),
            (f'Shared: Object {name_obj} & MISV-', '#9B59B6'),
            ('Shared: All Three', '#E67E22')
        ]
        shared_colors = ['#2ECC71', '#9B59B6', '#E67E22']

    words_to_plot = []
    word_vectors_to_plot = []
    word_colors = []

    for idx, word in enumerate(words_filtered):
        sentence_set = frozenset(word_to_sentences[word])

        if explanation_type == 'complete' and sentence_set == frozenset([1, 2]):
            continue

        color = color_map.get(sentence_set, '#5E6969')
        words_to_plot.append(word)
        word_vectors_to_plot.append(word_vectors_filtered[idx])
        word_colors.append(color)

    words_filtered = np.array(words_to_plot)
    word_vectors_filtered = np.array(word_vectors_to_plot)

    if len(words_filtered) == 0:
        raise ValueError("No words to plot after filtering")

    Z = linkage(word_vectors_filtered, method=method, metric=metric)
    plt.figure(figsize=(18, 10))

    dend = dendrogram(
        Z,
        labels=words_filtered,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=0
    )

    # Color the leaf labels based on sentence membership
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        word = lbl.get_text()
        if word in words_filtered:
            word_idx = list(words_filtered).index(word)
            color = word_colors[word_idx]
            lbl.set_color(color)
            # Make shared words bold
            if color in shared_colors:
                lbl.set_fontweight('bold')
            else:
                lbl.set_fontweight('normal')

    plt.suptitle(f'Prediction on context {context}: {class_name}')
    plt.title(f'Hierarchical Clustering: {explanation_type.capitalize()} Explanation')
    plt.xlabel("Words", fontsize=12, fontweight='bold')
    plt.ylabel(f"Distance ({method.capitalize()} Linkage)", fontsize=12, fontweight='bold')

    legend_elements = []
    for label, color in legend_labels:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10,
                       label=label, markeredgewidth=1.5,
                       markeredgecolor='black')
        )

    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("{0}hierarchical_clustering_{1}.pdf".format(output_dir, name_obj), dpi=300)
    plt.show()