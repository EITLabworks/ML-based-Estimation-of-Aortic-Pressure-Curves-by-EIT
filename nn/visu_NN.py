import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import numpy as np


def get_depth_draw(depth, bLogScale=True):
    if depth== 1:
        depth=2
    elif bLogScale:
        depth = np.emath.logn(1.1, depth)
    elif depth> 500:
        depth= depth//8
    elif depth>200:
        depth= depth//3
    return depth

def get_width_draw(width):
    if width == 1:
        width = 2
    #   elif depth> 500:
    #      depth= depth//8
    # elif depth>200:
    #    depth= depth//3
    else:
        width = np.emath.logn(1.2, width)
    return width

def calc_bottom_left(xoffset, h, w, depth):
    # xoffset is left side, but in depth and heigth middle
    depth = get_depth_draw(depth)


    frontplane= [xoffset -depth//4, 0-depth//4]
    frontplane[1] = frontplane[1] - h//2
    return frontplane


def draw_3d_box(ax, bottom_left, width, height, depth, color, bSize, wreal, bText=True, bDepthLog=True):
    x, y = bottom_left
    # Vertices of the box
    front_bottom_left = np.array([x, y])
    front_bottom_right = np.array([x + width, y])
    front_top_left = np.array([x, y + height])
    front_top_right = np.array([x + width, y + height])
    depth_plot= depth
    depth= get_depth_draw(depth, bDepthLog)
    back_bottom_left = front_bottom_left + np.array([depth//2, depth//2])
    back_bottom_right = front_bottom_right + np.array([depth//2, depth//2])
    back_top_left = front_top_left + np.array([depth//2, depth//2])
    back_top_right = front_top_right + np.array([depth//2, depth//2])



    # Draw back face
    ax.fill([back_bottom_left[0], back_bottom_right[0], back_top_right[0], back_top_left[0]],
            [back_bottom_left[1], back_bottom_right[1], back_top_right[1], back_top_left[1]], color, alpha=1, edgecolor="black")
    # Draw front face
    ax.fill([front_bottom_left[0], front_bottom_right[0], front_top_right[0], front_top_left[0]],
            [front_bottom_left[1], front_bottom_right[1], front_top_right[1], front_top_left[1]], color, alpha=1, edgecolor="black")

    # Draw right face
    ax.fill([front_bottom_right[0], back_bottom_right[0], back_top_right[0], front_top_right[0]],
            [front_bottom_right[1], back_bottom_right[1], back_top_right[1], front_top_right[1]], color, alpha=1, edgecolor="black")
    # Draw top face
    ax.fill([front_top_left[0], back_top_left[0], back_top_right[0], front_top_right[0]],
            [front_top_left[1], back_top_left[1], back_top_right[1], front_top_right[1]], color, alpha=1, edgecolor="black")

    if bSize:
        #todo
        front_bottom_mid =(front_bottom_right[0]- front_bottom_left[0])//2+ front_bottom_left[0]
        if bText:
            ax.text(front_bottom_mid, front_bottom_left[1]-10,str(depth_plot)+" x "+str(height)+" x "+str(wreal), horizontalalignment="center")


def get_box_whd(dim_string):
    # list of style(  [`(None,", "100,", "50,", "5)"]
    if len(dim_string)==1:
        return 1,1,1,1
    r = []
    for k in range(1,len(dim_string)):
        number = re.match(r"\d{1,4}",dim_string[k] )
        num = int(number.group())
        r.append(num)
    if len(r)==1:
        w = r[0]
        if r[0]>1000:
            return r[0]//50, 1,1,r[0]
        if r[0]> 150:
            return r[0]//20, 1,1,r[0]
        else:
            return r[0]//10, 1,1, r[0]
    elif len(r)==2:
        return 1, r[0], r[1], 1
    else:
        return r[2]/2,r[0], r[1], r[2]
def check_empty_line(l):
    for j in l:
        if j!=" ":
            return False
    return True


def draw_description_boxes(used_layers, x_offset, ax, bText=True):
    description_offset_y= 5+len(used_layers)*10
    description_offset_x= x_offset-40
    for t, c in used_layers.items():
        draw_3d_box(ax, [description_offset_x, description_offset_y], 5, 5, 5,c , False, 0, bText=bText, bDepthLog=False)
        if bText:
            ax.text(description_offset_x+15, description_offset_y+2.5, t, horizontalalignment="left", verticalalignment="center")
        description_offset_y -= 10


def draw_description_boxes_split(used_layers, x_offset, ax, bText=True):
    if len(used_layers)%2==0:
        description_offset_y= len(used_layers)//2*10
        top_values = len(used_layers)//2
    else:
        description_offset_y= (len(used_layers)//2+1)*10
        top_values = len(used_layers) // 2+1

    description_offset_x= x_offset-50
    counter=0
    for t, c in used_layers.items():
        counter+=1
        draw_3d_box(ax, [description_offset_x, description_offset_y], 5, 5, 5,c , False, 0, bText=bText, bDepthLog=False)
        if bText:
            ax.text(description_offset_x+15, description_offset_y+2.5, t, horizontalalignment="left", verticalalignment="center")
        description_offset_y -= 10
        if counter==top_values:
            description_offset_y-=15

def plot_model_structure(model, input_dataname, output_dataname="Output",xspacing=5,  bText=True, bArrows=True):
    # Create a figure

    model.summary()
    # Parse the model summary
    layer_info = []
    model.summary(print_fn=lambda x: layer_info.append(x))

    # Remove the first two lines which are model info, and the last two lines which are summary info
    layer_info = layer_info[4:-5]
    fig, ax = plt.subplots(figsize=(18,12))


    # Initialize parameters for plotting
    y_start= 0
    x_offset = 0
    if bText:
        ax.text(x_offset-50,y_start, input_dataname, verticalalignment="center",horizontalalignment="right", fontsize=16)
    if bArrows:
        ax.arrow(x_offset-48,y_start,12,0, color="black" ,    head_width = 3,    width = 0.2, head_length=3)
    spacing = xspacing

    used_layers  = {}
    for layer in layer_info:
        empty = check_empty_line(layer)
        if empty:
            continue
        parts = layer.split()
        if len(parts)==1:
            continue
        layer_type = parts[1][1:-1]     # ausschneiden der klammmer

        layer_shape = parts[-1]
        w, h,d,wreal  =get_box_whd(parts[2:-1])
        bSize= False
        print(layer_type)
        if 'Conv2D' in layer_type:
            color = 'steelblue'
            bSize= True
        elif 'InputLayer' in layer_type:
            color = 'lightblue'
            bSize= True
        elif 'Dense' in layer_type:
            color = 'teal'
            bSize= True
        elif "MaxPooling" in layer_type:
            color="brown"
        elif "Activation" in layer_type:
            color="goldenrod"
        elif "Dropout" in layer_type:
            color="peru"

        elif "Flatten" in layer_type:
            color="rosybrown"
            bSize= True

        elif "Ba" in layer_type:
            layer_type = "BatchNormalization"
            color="lightgray"
        else:
            color = 'lightgray'
        if layer_type not in used_layers:
            used_layers.update({layer_type:color})
        bottomleft =calc_bottom_left(x_offset, h,w, d)

        y_offset= -h//2
        print(y_offset)
        draw_3d_box(ax, bottomleft, w,h,d, color, bSize, wreal, bText=bText)
        x_offset= x_offset+w+spacing

    if output_dataname!= "Output":
        if bArrows:
            ax.arrow(x_offset,y_start,12,0, color="black" ,  head_width = 3,    width = 0.2, head_length=3)
        if bText:
            ax.text(x_offset+18,y_start, output_dataname, verticalalignment="center",horizontalalignment="left", fontsize=16)



    draw_description_boxes_split(used_layers, x_offset,ax, bText=bText)

        # Draw the rectangle for the layer
    #    rect = Rectangle((x_offset, y_offset), layer_width, layer_height, linewidth=1, edgecolor='black',
    ##                     facecolor=color)
    #    ax.add_patch(rect)

        # Add text for the layer type and shape
     #   ax.text(x_offset + layer_width / 2, y_offset + layer_height / 2, layer_type, fontsize=10, ha='center',
     #           va='center')
     #   ax.text(x_offset + layer_width / 2, y_offset - spacing / 2, layer_shape, fontsize=8, ha='center', va='center')

        # Move to the next layer position
    #    y_offset -= (layer_height + spacing)

    # Adjust the plot limits and add labels
  #  ax.set_xlim(0, x_offset + layer_width + 1)
  # ax.set_ylim(y_offset - 1, layer_height + 1)
 #   ax.set_xticks([])
    plt.tight_layout()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid()

    plt.show()

from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import argparse
from tensorflow.keras.models import Model

# ------------------------------------------------------------------------------------------------------------------- #
def stride_pattern(input_cha_num):
    """
    Calculates the used stride pattern depending on the number of input channels (EIT Indices)
    :param input_cha_num: number of input channels (EIT Indices)
    :return: vector with the for stride parameters
    """
    if input_cha_num > 700:
        p = [4, 4, 4, 4]
    elif input_cha_num > 400:
        p = [4, 4, 4, 2]
    elif input_cha_num > 200:
        p = [2, 4, 4, 2]
    else:
        p = [2, 2, 2, 2]
    return p

def model(input_shape=(128, 1024, 1), latent_dim=3, kernel=5, filter1=8, filter2=8, filter3=12, actiConv="elu",
          actiDense="elu", actiOutput="relu", factor=3, bDropout=False, numDrop=0.1, bWeightingLayer=False, bBatch=False ):

    pattern = stride_pattern(input_shape[1])  # usually 4,4,4,4
    print(pattern)

    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(int(filter1), kernel, strides=(2, pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)

    x = Conv2D(int(filter2), kernel, strides=(2, pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)
    x = Conv2D(int(filter3), kernel, strides=(1, pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(1, pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)

    x = Flatten()(x)
    x = Dense(int(factor)*latent_dim, activation=actiDense)(x)# elu #hinzugefügt
    x = Dense(2*latent_dim, activation=actiDense)(x)# elu

    # x = Dense(latent_dim, activation="elu")(x)
    mapper_output = Dense(latent_dim, activation=actiOutput)(x)  ##linear

    return Model(mapper_input, mapper_output)


# Model Training ----------------------------------------------------------------------------- #
if __name__=="__main__":
    from nn.nn_models import model_selection
    eit_sample_len = 1024

    model_fct = model_selection["ktpart1"]
    m1 = model_fct(input_shape=(64, eit_sample_len, 1), latent_dim=42, ConfigParas=None)

    plot_model_structure(m1, "EIT-Data", "Aortic Parameters", xspacing=3, bText=True, bArrows=True)
    model = model(input_shape=(64, eit_sample_len, 1), latent_dim=40, kernel=5,
                  filter1=8, filter2=8, filter3=8, actiConv="elu", actiDense="selu", actiOutput="relu",
                  factor=3, bDropout=True, numDrop=0.5, bWeightingLayer=False, bBatch=False)
    model.summary()
    plot_model_structure(model, "EIT-Data", "Aortic Parameters", bText=False)

    import logging

    logging.warning("The layer name ")
    # Define your model (for demonstration purposes)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(name="BN"),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    # Plot the model structure
    plot_model_structure(model, "EIT-Data")

    x= np.arange(1,3000)

    for k in range(1,10):
        plt.plot(np.emath.logn(k, x), label = str(k))
    plt.plot(np.emath.logn(1.5, x), label=str(1.5))
    plt.plot(np.emath.logn(1.2, x), label=str(1.2))
    plt.legend()
    plt.grid()
    plt.show()