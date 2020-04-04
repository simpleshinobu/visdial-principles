from visdialch.encoders.lf_enhanced import LF_Enhanced_Encoder
from visdialch.encoders.lf_enhanced_withP1 import LF_Enhanced_withP1_Encoder
from visdialch.encoders.dict_encoder import Dict_Encoder


def Encoder(model_config, *args):
    name_enc_map = {
        'baseline_encoder': LF_Enhanced_Encoder,
        'baseline_encoder_withP1': LF_Enhanced_withP1_Encoder,
        'dict_encoder' : Dict_Encoder,
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
