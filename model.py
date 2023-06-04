import cv2

def load_model(weights_file, config_file):
    # Load YOLOv3-tiny model from the given weights file
    net = cv2.dnn.readNet(weights_file, config_file)

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers
