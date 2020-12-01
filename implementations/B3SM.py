import tensorflow as tf

from util.Config import IMAGE_WIDTH, IMAGE_HEIGHT
from util.layers.convolutions import conv_layer, deconv_layer
from util.layers.normalizations import batch_norm
from util.layers.pooling_layers import max_pooling
from util.layers.usim import usim_layer as USIM


class B3SM:

    def __init__(self, imgHolder, batch_size, numChannels=3):
        self.batch_size = batch_size
        self.imgHolder = imgHolder
        self.numChannels = numChannels

    def __conv__(self, input, input_channel, output_channel, name='layer'):
        layer = conv_layer(input, input_channel, output_channel, filter_size=3, name=name)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, keep_prob=1.0)
        layer = batch_norm(layer, output_channel, name=name)
        return layer

    def __deconv__(self, input, input_channel, output_channel, name='layer'):

        layer = deconv_layer(input, input_channel, output_channel, batch_size=self.batch_size, name=name)
        layer = tf.nn.dropout(layer, keep_prob=1.0)
        layer = batch_norm(layer, output_channel, name=name)
        return layer

    def __resi_block__(self, layer, channel, name):

        layer1 = self.__conv__(layer, channel, channel, name="%s_resi_block01" % name)
        layer2 = self.__conv__(layer1, channel, channel, name="%s_resi_block02" % name)
        layer3 = self.__conv__(layer2, channel, channel, name="%s_resi_block03" % name)
        layer4 = tf.add(layer, layer3)
        return layer4

    def __conv_resi_conv__(self, input, channel1, channel2, channel3, name='crc'):

        layer1 = self.__conv__(input, input_channel=channel1, output_channel=channel2, name='%s_crc1' % name)
        layer2 = self.__resi_block__(layer1, channel2, name='%s_crc2' % name)
        layer3 = self.__conv__(layer2, channel2, channel3, name='%s_crc3' % name)
        return layer3

    def __merge__(self, layer1, layer2):
        return tf.add_n([layer1, layer2])


    def structure(self):
        semi_prediction = self.__fusionBlock__(self.imgHolder, self.numChannels)
        semi_prediction = tf.split(semi_prediction, num_or_size_splits=2, axis=3)[1]
        semi_prediction = tf.concat([semi_prediction, semi_prediction, semi_prediction], axis=3)

        after_usim = USIM(semi_prediction, self.imgHolder, self.batch_size)

        pred, conv = self.__fusion2Block__(after_usim, 3)

        return pred, conv

    def __fusion2Block__(self, input, channel):
        layer011 = self.__conv__(input, channel, 64, name='layer1011')

        layer112 = self.__conv_resi_conv__(layer011, 64, 64, 64, name='layer1112')
        maxpool11 = max_pooling(layer112)

        layer113 = self.__conv_resi_conv__(maxpool11, 64, 128, 128, name='layer1113')
        maxpool12 = max_pooling(layer113)

        layer114 = self.__conv_resi_conv__(maxpool12, 128, 256, 256, name='layer1114')
        maxpool13 = max_pooling(layer114)

        layer115 = self.__conv_resi_conv__(maxpool13, 256, 512, 512, name='layer1115')
        maxpool14 = max_pooling(layer115)

        # bridge
        layer116 = self.__conv_resi_conv__(maxpool14, 512, 1024, 1024, name='layer1116')
        layer116 = self.__conv__(layer116, 1024, 512, name='upscaling005')

        # deconv
        upscaling4 = self.__conv_resi_conv__(layer116, 512, 512, 512, name='up04')
        upscaling4 = USIM(maxpool14, upscaling4, self.batch_size)
        upscaling4 = self.__conv__(upscaling4, 512, 256, name='upscaling004')

        upscaling3 = self.__conv_resi_conv__(upscaling4, 256, 256, 256, name="up03")
        upscaling3 = USIM(maxpool13, upscaling3, self.batch_size)
        upscaling3 = self.__conv__(upscaling3, 256, 128, name='upscaling003')


        upscaling2 = self.__conv_resi_conv__(upscaling3, 128, 128, 128, name="up02")
        upscaling2 = USIM(maxpool12, upscaling2, self.batch_size)
        upscaling2 = self.__conv__(upscaling2, 128, 64, name='upscaling002')

        conv = conv_layer(upscaling2, 64, 2, filter_size=1, name='conv')
        pred = tf.argmax(conv, axis=3, name="prediction")
        return pred, conv

    def __fusionBlock__(self, input, channel):
        layer011 = self.__conv__(input, channel, 64, name='layer011')

        layer112 = self.__conv_resi_conv__(layer011, 64, 64, 64, name='layer112')
        maxpool11 = max_pooling(layer112)

        layer113 = self.__conv_resi_conv__(maxpool11, 64, 128, 128, name='layer113')
        maxpool12 = max_pooling(layer113)

        layer114 = self.__conv_resi_conv__(maxpool12, 128, 256, 256, name='layer114')
        maxpool13 = max_pooling(layer114)

        layer115 = self.__conv_resi_conv__(maxpool13, 256, 512, 512, name='layer115')
        maxpool14 = max_pooling(layer115)

        # bridge
        layer116 = self.__conv_resi_conv__(maxpool14, 512, 1024, 1024, name='layer116')

        # deconv
        upscaling4 = self.__deconv__(layer116, 1024, 512, name='upscaling4')
        upscaling4 = self.__merge__(upscaling4, layer115)
        upscaling4 = self.__conv_resi_conv__(upscaling4, 512, 512, 512, name='up4')

        upscaling3 = self.__deconv__(upscaling4, 512, 256, name='upscaling3')
        upscaling3 = self.__merge__(upscaling3, layer114)
        upscaling3 = self.__conv_resi_conv__(upscaling3, 256, 256, 256, name="up3")

        upscaling2 = self.__deconv__(upscaling3, 256, 128, name="upscaling2")
        upscaling2 = self.__merge__(upscaling2, layer113)
        upscaling2 = self.__conv_resi_conv__(upscaling2, 128, 128, 128, name="up2")

        upscaling1 = self.__deconv__(upscaling2, 128, 64, name='upscaling1')
        upscaling1 = self.__merge__(upscaling1, layer112)
        upscaling1 = self.__conv_resi_conv__(upscaling1, 64, 64, 64, name='up1')

        conv = conv_layer(upscaling1, 64, 2, filter_size=1, name='conv_tmp')
        return conv





if __name__ == '__main__':
    imgHolder = tf.placeholder(dtype=tf.float32, shape=[10, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    prediction, logits = B3SM(imgHolder, 10).structure()

    print(prediction.shape)
    print(logits.shape)


