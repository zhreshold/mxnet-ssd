nvidia-docker run -it --rm \
-v /home/oper/david/mxnet-ssd/model:/mxnet/example/ssd/model \
-v /home/oper/david/mxnet-ssd/detect:/mxnet/example/ssd/detect \
-v /home/oper/david/mxnet-ssd/evaluate:/mxnet/example/ssd/evaluate \
-v /home/oper/david/mxnet-ssd/data:/mxnet/example/ssd/data \
-v /home/oper/david/mxnet-ssd/scripts:/mxnet/example/ssd/scripts \
mxnet/ssd:david
