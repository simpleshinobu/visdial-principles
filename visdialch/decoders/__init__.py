from visdialch.decoders.disc_by_round import Disc_by_round_Decoder
from visdialch.decoders.disc_qt import Disc_qt_Decoder



def Decoder(model_config, *args):
    name_dec_map = {
                    'disc_by_round': Disc_by_round_Decoder,
                    'disc_qt': Disc_qt_Decoder,
                    }
    return name_dec_map[model_config["decoder"]](model_config, *args)
