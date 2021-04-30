% modelfile = 'cifarResNet.onnx';
% classes = ["airplane" "automobile" "bird" "cat" "dee" "dog" "frog" "horse" "ship" "truck"];
% net = importONNXNetwork(modelfile,'OutputLayerType','classification','Classes',classes)

modelfile = 'alexnet_onnx.onnx';
alexnet = importONNXNetwork(modelfile,'OutputLayerType','classification');

params  = importONNXFunction(modelfile,'alexnetFcn');
% analyzeNetwork(alexnet)
camera = webcam;
cam_img = snapshot(camera);
resize_img = imresize(cam_img, [224,224]);
trans_img = permute(resize_img,[1,2,3]);
input_img = zeros(224,224,3,1);
input_img(:,:,:,1) = trans_img;

output_net = alexnetFcn(input_img,params);

%%

modelfile = 'resnet_model.onnx';
resnet_tf = importONNXNetwork(modelfile,'OutputLayerType','classification');

params  = importONNXFunction(modelfile,'resnet_tf');
output_net = resnet_tf(input_img,params);
