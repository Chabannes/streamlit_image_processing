from helper import *

import streamlit as st
from PIL import Image
from io import BytesIO
import webcolors
import Cython


# data analysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from importlib import reload
import seaborn as sns

# modeling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# from pyngrok import ngrok
# public_url = ngrok.connect(port='8080')
# print(public_url)


def main():

    # front end elements of the web page
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:white;text-align:center;">Streamlit Image Processing and Object Detection</h1>
    </div>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)


    # LOAD ORIGINAL IMAGE

    st.subheader('Upload your image')
    st.text('Welcome! This is a simple image processing app using Streamlit and Heroku.')
    st.text('Your image must be in a 200x200 png format.\nYou can find such pictures at https://github.com/Chabannes/streamlit_image_processing/data')
    uploaded_file = st.file_uploader("Choose an image...", type="png")
    if uploaded_file is not None:

        # display original image
        st.title('Your original image')
        ori_img = Image.open(uploaded_file)
        #ori_img = ori_img.resize((220, 220))
        X = np.array(ori_img.getdata())

        ori_img_size = imageByteSize(ori_img)
        ori_img_n_colors = len(set(ori_img.getdata()))
        ori_img_total_variance = sum(np.linalg.norm(X - np.mean(X, axis=0), axis=1) ** 2)
        ori_pixels = X.reshape(*ori_img.size, -1)

        st.subheader('Size of the original image: %s KB' % (round(ori_img_size, 3)))
        st.subheader('Number of different colors: %s' % (ori_img_n_colors))
        st.image(ori_img, caption='Original Image', use_column_width=True)

        st.title("Choose the task you want to perform")
        st.text('You can either apply a compression to your image or perform object detection. The yolo \nmodel used for object'
                'detection being too heavy for Github, this functionality is not \navailable if you are using a public URL to run this app.')
        task = st.selectbox('Task to perform', ('', 'Apply Compression', 'Perform Object Detection'))

        if task == 'Apply Compression':

            st.header('Two types of compression are available\n')
            st.subheader('The color based reduction:')
            st.text('The image you uploaded contains %s different colors. The idea is to reduce the number \nof colors using'
                    'Kmeans clustering. Each of the %s different colors will be assigned \nto the closest color among the k '
                    'new colors, reducing drastically the new image size.\n' %(ori_img_n_colors, ori_img_n_colors))

            st.subheader('The Principal Components based reduction:')
            st.text('For a Principal Components compression, the idea is to switch to a new orthogonal \nbasis. The '
                    'first component of this new basis is the one explaining the most variance \nin the data. The '
                    'more components, the more variance (i.e. information) get restored from \nthe original data.')

            st.header('Choose your compression type')
            comp_type = st.selectbox('Compression Type', ('', 'Color Reduced', 'Principal Component Reduced'))

            st.subheader(comp_type)

            if comp_type == 'Color Reduced':

                st.text('You have to choose the K value in the Kmeans clustering algorithm. In other words, you \nset '
                        'the new number of colors that the initial %s different colors will be reduced to.' %(ori_img_n_colors))

                n_clusters = st.slider('How Many Colors ? ', 1, 80)

                # KMEANS CLUSTERING

                kmeans = KMeans(n_clusters=n_clusters,
                               n_jobs=-1,
                               random_state=123).fit(X)
                kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Red', 'Green', 'Blue'])
                kmeans_df["Color Name"] = list(map(get_colour_name, np.uint8(kmeans.cluster_centers_)))

                new_pixels = replaceWithCentroid(kmeans, ori_img)

                def plotImage(img_array, size):
                    reload(plt)
                    plt.imshow(np.array(img_array / 255).reshape(*size))
                    plt.axis('off')
                    return plt

                WCSS = kmeans.inertia_
                BCSS = calculateBCSS(X, kmeans)
                exp_var = 100 * BCSS / (WCSS + BCSS)

                st.subheader("Image Size: {:.2f} KB / Initially {:.2f} KB".format(imageByteSize(new_pixels), ori_img_size))
                st.subheader("Number of Colors: %s / Initially %s" %(n_clusters, ori_img_n_colors))
                st.subheader("Explained Variance: {:.3f}%".format(exp_var))

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(np.array(new_pixels / 255).reshape(*new_pixels.shape))
                ax.axis('off')
                st.write(fig)

                st.text('You can also display the evolution of the compression with the number of colors but\nthis is a '
                        'very time consuming computation.')

                show_color_evol = st.button("Display the evolution of compression with the number of colors")
                if show_color_evol:

                    range_k_clusters = (2, 21)

                    kmeans_result = []
                    for k in range(*range_k_clusters):
                        # CLUSTERING
                        kmeans = KMeans(n_clusters=k,
                                        n_jobs=-1,
                                        random_state=123).fit(X)

                        # REPLACE PIXELS WITH ITS CENTROID
                        #new_pixels = replaceWithCentroid(kmeans, ori_img)

                        new_pixels = []
                        for label in kmeans.labels_:
                            pixel_as_centroid = list(kmeans.cluster_centers_[label])
                            new_pixels.append(pixel_as_centroid)
                        new_pixels = np.array(new_pixels).reshape(*ori_img.size, -1)

                        # EVALUATE
                        WCSS = kmeans.inertia_
                        BCSS = calculateBCSS(X, kmeans)
                        exp_var = 100 * BCSS / (WCSS + BCSS)

                        metric = {
                            "No. of Colors": k,
                            "Centroids": list(map(get_colour_name, np.uint8(kmeans.cluster_centers_))),
                            "Pixels": new_pixels,
                            "WCSS": WCSS,
                            "BCSS": BCSS,
                            "Explained Variance": exp_var,
                            "Image Size (KB)": imageByteSize(new_pixels)
                        }

                        kmeans_result.append(metric)
                    kmeans_result = pd.DataFrame(kmeans_result).set_index("No. of Colors")

                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    for ax, metric in zip(axes.flat, kmeans_result.columns[2:]):
                        sns.lineplot(x=kmeans_result.index, y=metric, data=kmeans_result, ax=ax)

                        if metric == "WCSS":
                            y_val = 0
                        elif metric == "BCSS":
                            y_val = ori_img_total_variance
                        elif metric == "Explained Variance":
                            y_val = 100
                        elif metric == "Image Size (KB)":
                            y_val = ori_img_size

                        ax.axhline(y=y_val, color='k', linestyle='--', label="Original Image")
                        ax.set_xticks(kmeans_result.index[::2])
                        ax.ticklabel_format(useOffset=False)
                        ax.legend()
                    plt.tight_layout()
                    fig.suptitle("METRICS BY NUMBER OF COLORS", size=25, y=1.03, fontweight="bold")
                    st.write(fig)

            if comp_type == 'Principal Component Reduced':

                @st.cache
                def load():
                    res = []
                    cum_var = []
                    X_t = np.transpose(X)
                    for channel in range(3):
                        # SEPARATE EACH RGB CHANNEL
                        pixel = X_t[channel].reshape(*ori_pixels.shape[:2])
                        # PCA
                        pca = PCA(random_state=123)
                        pixel_pca = pca.fit_transform(pixel)
                        pca_dict = {
                            "Projection": pixel_pca,
                            "Components": pca.components_,
                            "Mean": pca.mean_
                        }
                        res.append(pca_dict)

                        # EVALUATION
                        cum_var.append(np.cumsum(pca.explained_variance_ratio_))

                    cum_var_df = pd.DataFrame(np.array(cum_var).T * 100,
                                              index=range(1, pca.n_components_ + 1),
                                              columns=["Explained Variance by Red",
                                                       "Explained Variance by Green",
                                                       "Explained Variance by Blue"])
                    cum_var_df["Explained Variance"] = cum_var_df.mean(axis=1)
                    return cum_var_df, res, pca

                cum_var_df, res, pca = load()

                n_components = st.slider('How Many Principal Components ? ', 2, 100)
                temp_res = []
                for channel in range(3):
                    pca_channel = res[channel]
                    pca_pixel = pca_channel["Projection"][:, :n_components]
                    pca_comp = pca_channel["Components"][:n_components, :]
                    pca_mean = pca_channel["Mean"]
                    compressed_pixel = np.dot(pca_pixel, pca_comp) + pca_mean
                    temp_res.append(compressed_pixel.T)
                compressed_image = np.transpose(temp_res)

                pca_comp_size = imageByteSize(compressed_image)
                exp_var_pca = cum_var_df["Explained Variance"][n_components]

                # st.subheader("Image Size: {:.2f} KB / Initially {:.2f} KB".format(imageByteSize(new_pixels), ori_img_size))

                st.subheader("Image Size: {:.2f} KB / Initially {:.2f} KB".format(pca_comp_size, ori_img_size))
                st.subheader("Explained Variance: {:.2f}%".format(exp_var_pca))

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(np.array(compressed_image / 255).reshape(*compressed_image.shape))
                ax.axis('off')
                st.write(fig)

                st.text('You can also display the evolution of the compression with the number of principal \n'
                        'components but this is a very time consuming computation.')

                show_comp_evol = st.button("Display the evolution of compression with the number of components")
                if show_comp_evol:

                    pca_results = []
                    for n in range(1, pca.n_components_ + 1):
                        # SELECT N-COMPONENTS FROM PC
                        temp_res = []
                        for channel in range(3):
                            pca_channel = res[channel]
                            pca_pixel = pca_channel["Projection"][:, :n]
                            pca_comp = pca_channel["Components"][:n, :]
                            pca_mean = pca_channel["Mean"]
                            compressed_pixel = np.dot(pca_pixel, pca_comp) + pca_mean
                            temp_res.append(compressed_pixel.T)
                        compressed_image = np.transpose(temp_res)

                        pca_dict = {
                            "n": n,
                            "Pixels": compressed_image,
                            "Explained Variance": cum_var_df["Explained Variance"][n],
                            "Image Size (KB)": imageByteSize(compressed_image),
                            "No. of Colors": len(np.unique(np.uint8(compressed_image).reshape(-1, 3), axis=0))
                        }

                        pca_results.append(pca_dict)

                    pca_results = pd.DataFrame(pca_results).set_index("n")

                    line_colors = "ygr"
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for ax, metric in zip(axes, pca_results.columns[1:]):
                        sns.lineplot(x=pca_results.index, y=metric, data=pca_results, ax=ax)
                        ax.set_xlabel("No. of Principal Components")

                        if metric == "Explained Variance":
                            lookup_n_var = []
                            for idx, exp_var in enumerate([90, 95, 99]):
                                lookup_n = pca_results[pca_results[metric] >= exp_var].index[0]
                                lookup_n_var.append(lookup_n)
                                ax.axhline(y=exp_var, color=line_colors[idx], linestyle='--',
                                           label="{}% Explained Variance (n = {})".format(exp_var, lookup_n))
                                ax.plot(lookup_n, exp_var, color=line_colors[idx], marker='x', markersize=8)
                                ax.set_ylabel("Cumulative Explained Variance (%)")
                            ax.legend()
                            continue
                        elif metric == "Image Size (KB)":
                            y_val = ori_img_size
                            line_label = "n = {} (Size: {:.2f} KB)"
                        elif metric == "No. of Colors":
                            y_val = ori_img_n_colors
                            line_label = "n = {} (Colors: {})"

                        ax.axhline(y=y_val, color='k', linestyle='--', label="Original Image")
                        for idx, n_components in enumerate(lookup_n_var):
                            lookup_value = pca_results.loc[n_components, metric]
                            ax.axvline(x=n_components, color=line_colors[idx], linestyle='--',
                                       label=line_label.format(n_components, lookup_value))
                            ax.plot(n_components, lookup_value, color=line_colors[idx], marker='x', markersize=8)
                        ax.legend()
                    plt.tight_layout()
                    fig.suptitle("METRICS BY NUMBER OF PRINCIPAL COMPONENTS", size=30, y=1.07, fontweight="bold")
                    st.write(fig)

        if task == 'Perform Object Detection':

            try:
                model = load_model('yolo.h5')

                class_threshold = st.slider('Level of confidence', 0.1, 0.9)
                image_size = 416, 416  # expected input shape for the model
                anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
                detection(ori_img, model, anchors, class_threshold, image_size)
            except:
                st.text("Can't import the model needed for object detection. The model can't be loaded if \nyou are"
                        " running this app using a public URL (too heavy)")



def imageByteSize(img):
    img_file = BytesIO()
    image = Image.fromarray(np.uint8(img))
    image.save(img_file, 'png')
    return img_file.tell()/1024


def replaceWithCentroid(kmeans, ori_img):
    new_pixels = []
    for label in kmeans.labels_:
        pixel_as_centroid = list(kmeans.cluster_centers_[label])
        new_pixels.append(pixel_as_centroid)
    new_pixels = np.array(new_pixels).reshape(*ori_img.size, -1)
    return new_pixels


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name


def calculateBCSS(X, kmeans):
    _, label_counts = np.unique(kmeans.labels_, return_counts=True)
    diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis=0), axis=1) ** 2
    return sum(label_counts * diff_cluster_sq)


if __name__ == '__main__':
    main()

