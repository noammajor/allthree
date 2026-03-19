config = {
    "path_save": "./output_model/DiscreteJEPA/",
    #learning rates (adjusted for AdamW optimizer - further reduced for stability)
    "lr": 8e-5,  # Increased: encoder was barely learning at 1e-4
    "end_lr": 5e-6,
    "num_epochs": 1501,
    "ema_momentum" : 0.996,  # Slightly increased for smoother target
    "codebook_lr" : 5e-4,  # Increased - codebook needs to move faster in 128-dim space
    "weight_decay" : 3e-3,  # Increased from 5e-5 for better regularization
    "perplexity_loss_weight": 0.5,  # Increased from 1.2 (will be used properly)
    #added
    "lr_pred": 6e-5,  # Matched to encoder: prevents predictor from dominating
    "weight_decay_pred" : 1e-4,  # Increased from 1e-6

    #masking
    "mask_ratio" : 0.2,
    "masking_type" : "multi_block",
    "num_blocks" : 3,

    #encoder
    "num_semantic_tokens" : 8,
    "encoder_embed_dim" : 64,
    "nhead" : 8,
    "num_encoder_layers" : 3,
    "mlp_ratio" : 4.0,
    "qkv_bias" : True,
    "qk_scale" : None,
    "drop_rate" : 0.00,
    "attn_drop_rate" : 0.00,
    "kernel_size" : 6,
    "encoder_kernel_size" : 6,
    "embed_bias" : True,
    "encoder_embed_bias" : True,
    "codebook_size" : 256,
    "commitment_cost" : 0.25,  # Stronger pull toward codebook usage
    "vq_ema_decay" : 0.99,  # EMA decay for codebook updates (teacher codebook)
    "patch_size": 24,
    "patch_size_forcasting": 24,

    #predictor
    "predictor_embed_dim": 64,
    "predictor_nhead" : 4,
    "predictor_num_layers" : 2,

    #data
    "checkpoint_save" : 5000,
    "checkpoint_print": 30,
    "ratio_patches" : 24,
    "batch_size" : 64,
    #"batch_size" : 64,

    # Loader
    "clip_grad": 2.0,  # Increased from 0.5: encoder was gradient-starved through 16 layers
    "warmup_ratio": 0.50,
    "ipe_scale": 1.25,

    #weights for loss terms
    "lambda_weights" : {
        "P2P": 1.0,
        "S2P": 1.0,
        "P2S":1.0,
    },
    "preplexity_coeff": 0.2,
    "token_diversity": 0.15,
    "vigreg_patches": 0.00,
    "decorr_coeff": 0.0,
    "vigreg_var": 1.5,
    "vigreg_covar": 0.01,
    "vigreg_token": 0.25,
    "grounding_coeff": 0.10,
    "beta_vq" : 1.0,
    "vq_warmup": 0.01,
    "val_prec": 0.2,
    "test_prec": 0.1,
    "timestampcols": ["date"] * 94,
    "input_variables" : [
        ["0", "1", "2", "3"], ["4", "5", "6", "7"], ["8", "9", "10", "11"], ["12", "13", "14", "15"],
        ["16", "17", "18", "19"], ["20", "21", "22", "23"], ["24", "25", "26", "27"], ["28", "29", "30", "31"],
        ["32", "33", "34", "35"], ["36", "37", "38", "39"], ["40", "41", "42", "43"], ["44", "45", "46", "47"],
        ["48", "49", "50", "51"], ["52", "53", "54", "55"], ["56", "57", "58", "59"], ["60", "61", "62", "63"],
        ["64", "65", "66", "67"], ["68", "69", "70", "71"], ["72", "73", "74", "75"], ["76", "77", "78", "79"],
        ["80", "81", "82", "83"], ["84", "85", "86", "87"], ["88", "89", "90", "91"], ["92", "93", "94", "95"],
        ["96", "97", "98", "99"], ["100", "101", "102", "103"], ["104", "105", "106", "107"], ["108", "109", "110", "111"],
        ["112", "113", "114", "115"], ["116", "117", "118", "119"], ["120", "121", "122", "123"], ["124", "125", "126", "127"],
        ["128", "129", "130", "131"], ["132", "133", "134", "135"], ["136", "137", "138", "139"], ["140", "141", "142", "143"],
        ["144", "145", "146", "147"], ["148", "149", "150", "151"], ["152", "153", "154", "155"], ["156", "157", "158", "159"],
        ["160", "161", "162", "163"], ["164", "165", "166", "167"], ["168", "169", "170", "171"], ["172", "173", "174", "175"],
        ["176", "177", "178", "179"], ["180", "181", "182", "183"], ["184", "185", "186", "187"], ["188", "189", "190", "191"],
        ["192", "193", "194", "195"], ["196", "197", "198", "199"], ["200", "201", "202", "203"], ["204", "205", "206", "207"],
        ["208", "209", "210", "211"], ["212", "213", "214", "215"], ["216", "217", "218", "219"], ["220", "221", "222", "223"],
        ["224", "225", "226", "227"], ["228", "229", "230", "231"], ["232", "233", "234", "235"], ["236", "237", "238", "239"],
        ["240", "241", "242", "243"], ["244", "245", "246", "247"], ["248", "249", "250", "251"], ["252", "253", "254", "255"],
        ["256", "257", "258", "259"], ["260", "261", "262", "263"], ["264", "265", "266", "267"], ["268", "269", "270", "271"],
        ["272", "273", "274", "275"], ["276", "277", "278", "279"], ["280", "281", "282", "283"], ["284", "285", "286", "287"],
        ["288", "289", "290", "291"], ["292", "293", "294", "295"], ["296", "297", "298", "299"], ["300", "301", "302", "303"],
        ["304", "305", "306", "307"], ["308", "309", "310", "311"], ["312", "313", "314", "315"], ["316", "317", "318", "319"],
        # ETTh1 (2 groups of 4)
        ["HUFL", "HULL", "MUFL", "MULL"], ["LUFL", "LULL", "OT", "HUFL"],
        # ETTh2 (2 groups of 4)
        ["HUFL", "HULL", "MUFL", "MULL"], ["LUFL", "LULL", "OT", "HUFL"],
        # ETTm1 (2 groups of 4)
        ["HUFL", "HULL", "MUFL", "MULL"], ["LUFL", "LULL", "OT", "HUFL"],
        # ETTm2 (2 groups of 4)
        ["HUFL", "HULL", "MUFL", "MULL"], ["LUFL", "LULL", "OT", "HUFL"],
        # Weather (6 groups of 4; last group repeats "1","2","3" to fill 4th slot)
        ["1", "2", "3", "4"], ["5", "6", "7", "8"], ["9", "10", "11", "12"],
        ["13", "14", "15", "16"], ["17", "18", "19", "20"], ["OT", "1", "2", "3"],
    ],
    "path_data": (
        ["./data/electricity.csv"] * 80 +
        ["./data/ETTh1.csv"] * 2 +
        ["./data/ETTh2.csv"] * 2 +
        ["./data/ETTm1.csv"] * 2 +
        ["./data/ETTm2.csv"] * 2 +
        ["./data/weather.csv"] * 6
    ),
    "chunk_size": 128,
    # forcasting
    "epoch_t" : 250,
    "context_t": 24,
    "horizon_t": 4,
    "input_variables_forcasting": [
        ["1", "2", "3", "4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
    ],
    "val_prec_forcasting": 0.1,
    "test_prec_forcasting": 0.2,
    "timestampcols_forcasting": ['date'],
    "path_data_forcasting": ["./data/weather.csv"],
    "patches_to_forcast": 8,
    "patches_size_forecasting": 32,
    "lr_forcasting": 1e-3,
    "affine_revin": True
}