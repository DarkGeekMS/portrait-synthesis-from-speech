import numpy as np
import pickle

# function to save labels for hair color and eye color for images in images folder
# class1 : eye color      : black 0 , blue 1 , green 2 ,  brown 3
# class2 : Bags Under Eye : ON 1 , OFF 0 
# class3 : hair color     : black 0 , blonde 1 , Brown 2 , Gray 3 
# class4 : hair Length    : Short 0 , Tall 1.
# class5 : hair Style     : Straight 0 ,  Curly 1 ,  Receding Hairline 2 , Bald 3 , with Bangs 4.  
# class6 : Cheeks      : Normal 0 , Rosy     1 
# class7 : face Shape  : Oval   0 , Circular 1 
# class8 : Eyebrows    : Arched 0 , Bushy    1
# class9 : Ears        : Small  0 , Big      1
# class10: Cheekbones  : LOW    0 , High     1 
# class11: Double Chin : OFF    0 , ON       1
# class12: Facial Hair : None   0 , Goatee   1 , Mustache 2 , Beard 3 
# class13: Headwear    : OFF    0 , ON       1
# class14: Makeup      : None   0 , Slight   1 , Heavy    2
# class15: Lipstick    : OFF    0 , ON       1

def save():

    # read latent vectors
    latent_vectors = []
    for i in range (5700,6701):
        latent_vector_path = 'images/seed' + str(i) + '.npy'
        latent_vectors.append(np.load(latent_vector_path))


    # to save the datasets
    Eye_Color = {}
    Bags_Under_Eye = {}
    Hair_Color = {}
    Hair_Length = {}
    Hair_style = {}
    Cheeks = {}
    Face_Shape = {}
    Eyebrows = {}
    Ears = {}
    Cheekbones = {}
    Double_Chin = {}
    Facial_Hair = {}
    Headwear = {}
    Makeup = {}
    Lipstick = {}


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------ DONE ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

    # holds the latent vectors for every feature in hair color
    black_hair  = [latent_vectors[0], latent_vectors[9], latent_vectors[14], latent_vectors[17], latent_vectors[19], latent_vectors[25], latent_vectors[30],
    latent_vectors[37], latent_vectors[40], latent_vectors[43], latent_vectors[44], latent_vectors[154], latent_vectors[163], latent_vectors[167],
    latent_vectors[170], latent_vectors[174], latent_vectors[214], latent_vectors[215], latent_vectors[237], latent_vectors[247], latent_vectors[291],
    latent_vectors[292], latent_vectors[378], latent_vectors[318], latent_vectors[399], latent_vectors[300], latent_vectors[263], latent_vectors[262],
    latent_vectors[248], latent_vectors[208], latent_vectors[204], latent_vectors[188], latent_vectors[185], latent_vectors[159], latent_vectors[151],
    latent_vectors[990], latent_vectors[809], latent_vectors[690], latent_vectors[578], latent_vectors[543], latent_vectors[517], latent_vectors[507],
    latent_vectors[983], latent_vectors[884], latent_vectors[848], latent_vectors[755], latent_vectors[648], latent_vectors[593], latent_vectors[592],
    latent_vectors[979], latent_vectors[912], latent_vectors[883], latent_vectors[867], latent_vectors[866], latent_vectors[498], latent_vectors[481],
    latent_vectors[975], latent_vectors[969], latent_vectors[947], latent_vectors[926], latent_vectors[906], latent_vectors[565], latent_vectors[544],
    latent_vectors[956], latent_vectors[577], latent_vectors[479]  ]

    brown_hair  = [latent_vectors[1], latent_vectors[3], latent_vectors[5], latent_vectors[7], latent_vectors[8],  latent_vectors[11], latent_vectors[12],
    latent_vectors[13], latent_vectors[16], latent_vectors[21], latent_vectors[22], latent_vectors[23], latent_vectors[26], latent_vectors[27],
    latent_vectors[29], latent_vectors[32], latent_vectors[34], latent_vectors[41], latent_vectors[42], latent_vectors[160], latent_vectors[161],
    latent_vectors[169], latent_vectors[189], latent_vectors[190], latent_vectors[195], latent_vectors[211], latent_vectors[216], latent_vectors[297],
    latent_vectors[304], latent_vectors[379], latent_vectors[380], latent_vectors[433], latent_vectors[445], latent_vectors[473], latent_vectors[497],
    latent_vectors[999], latent_vectors[996], latent_vectors[988], latent_vectors[968], latent_vectors[960], latent_vectors[914], latent_vectors[862],
    latent_vectors[998], latent_vectors[980], latent_vectors[958], latent_vectors[916], latent_vectors[785], latent_vectors[774], latent_vectors[760],
    latent_vectors[995], latent_vectors[985], latent_vectors[978], latent_vectors[951], latent_vectors[948], latent_vectors[915], latent_vectors[842],
    latent_vectors[973], latent_vectors[966], latent_vectors[961], latent_vectors[959], latent_vectors[748], latent_vectors[707], latent_vectors[702],
    latent_vectors[962], latent_vectors[944], latent_vectors[941], latent_vectors[935], latent_vectors[931], latent_vectors[909], latent_vectors[905]  ]

    gray_hair   = [latent_vectors[2], latent_vectors[6], latent_vectors[18], latent_vectors[24], latent_vectors[31], latent_vectors[33], latent_vectors[35],
    latent_vectors[38], latent_vectors[39], latent_vectors[155], latent_vectors[186], latent_vectors[193], latent_vectors[197], latent_vectors[202],
    latent_vectors[229], latent_vectors[254], latent_vectors[275], latent_vectors[273], latent_vectors[294], latent_vectors[296], latent_vectors[299],
    latent_vectors[376], latent_vectors[377], latent_vectors[388], latent_vectors[393], latent_vectors[407], latent_vectors[437], latent_vectors[440],
    latent_vectors[477], latent_vectors[480], latent_vectors[482], latent_vectors[492], latent_vectors[467], latent_vectors[394], latent_vectors[62],
    latent_vectors[994], latent_vectors[886], latent_vectors[858], latent_vectors[846], latent_vectors[819], latent_vectors[813], latent_vectors[796],
    latent_vectors[953], latent_vectors[829], latent_vectors[822], latent_vectors[772], latent_vectors[759], latent_vectors[758], latent_vectors[757],
    latent_vectors[933], latent_vectors[927], latent_vectors[871], latent_vectors[765], latent_vectors[739], latent_vectors[729], latent_vectors[722],
    latent_vectors[911], latent_vectors[891], latent_vectors[753], latent_vectors[720], latent_vectors[714], latent_vectors[703], latent_vectors[701],
    latent_vectors[699], latent_vectors[693], latent_vectors[678], latent_vectors[669], latent_vectors[657], latent_vectors[642], latent_vectors[566]  ] 

    blonde_hair = [latent_vectors[4], latent_vectors[10], latent_vectors[15], latent_vectors[20], latent_vectors[28], latent_vectors[36], latent_vectors[153],
    latent_vectors[162], latent_vectors[166], latent_vectors[209], latent_vectors[210], latent_vectors[217], latent_vectors[219], latent_vectors[239],
    latent_vectors[266], latent_vectors[280], latent_vectors[281], latent_vectors[293], latent_vectors[298], latent_vectors[314], latent_vectors[382],
    latent_vectors[404], latent_vectors[442], latent_vectors[463], latent_vectors[483], latent_vectors[494], latent_vectors[462], latent_vectors[460],
    latent_vectors[458], latent_vectors[419], latent_vectors[417], latent_vectors[366], latent_vectors[338], latent_vectors[286], latent_vectors[218],
    latent_vectors[987], latent_vectors[974], latent_vectors[970], latent_vectors[952], latent_vectors[849], latent_vectors[841], latent_vectors[793],
    latent_vectors[955], latent_vectors[869], latent_vectors[672], latent_vectors[599], latent_vectors[587], latent_vectors[533], latent_vectors[513],
    latent_vectors[964], latent_vectors[954], latent_vectors[942], latent_vectors[709], latent_vectors[597], latent_vectors[548], latent_vectors[514],
    latent_vectors[892]  ]




    # holds the latent vectors for every feature in hair style
    straight_hair   = [latent_vectors[0], latent_vectors[1], latent_vectors[3], latent_vectors[4], latent_vectors[5], latent_vectors[8], latent_vectors[9],
    latent_vectors[10], latent_vectors[11], latent_vectors[12], latent_vectors[13], latent_vectors[15], latent_vectors[16], latent_vectors[17],
    latent_vectors[18], latent_vectors[19], latent_vectors[20], latent_vectors[21], latent_vectors[22], latent_vectors[23], latent_vectors[24],
    latent_vectors[25], latent_vectors[26], latent_vectors[27], latent_vectors[31], latent_vectors[32], latent_vectors[33], latent_vectors[34],
    latent_vectors[35], latent_vectors[36], latent_vectors[41], latent_vectors[42], latent_vectors[44], latent_vectors[176], latent_vectors[190],
    latent_vectors[461], latent_vectors[582], latent_vectors[655], latent_vectors[675], latent_vectors[682], latent_vectors[711], latent_vectors[716],
    latent_vectors[473], latent_vectors[494], latent_vectors[546], latent_vectors[597], latent_vectors[628], latent_vectors[629], latent_vectors[658],
    latent_vectors[480], latent_vectors[512], latent_vectors[544], latent_vectors[584], latent_vectors[589], latent_vectors[599], latent_vectors[612],
    latent_vectors[513], latent_vectors[531], latent_vectors[536], latent_vectors[549], latent_vectors[568], latent_vectors[580], latent_vectors[588],
    latent_vectors[631], latent_vectors[657], latent_vectors[680], latent_vectors[685], latent_vectors[756], latent_vectors[786], latent_vectors[789],
    latent_vectors[719], latent_vectors[743], latent_vectors[763], latent_vectors[793], latent_vectors[802], latent_vectors[807], latent_vectors[832]   ]

    curly_hair      = [latent_vectors[7], latent_vectors[147], latent_vectors[152], latent_vectors[486], latent_vectors[185], latent_vectors[70],
    latent_vectors[72], latent_vectors[378], latent_vectors[470], latent_vectors[475], latent_vectors[523], latent_vectors[590], latent_vectors[602],
    latent_vectors[550], latent_vectors[654], latent_vectors[677], latent_vectors[731], latent_vectors[748], latent_vectors[751], latent_vectors[791],
    latent_vectors[564], latent_vectors[678], latent_vectors[714], latent_vectors[775], latent_vectors[811], latent_vectors[834], latent_vectors[864],
    latent_vectors[887], latent_vectors[899], latent_vectors[954], latent_vectors[975]  ]


    rec_hair        = [latent_vectors[2], latent_vectors[6], latent_vectors[9], latent_vectors[14], latent_vectors[21], latent_vectors[28], latent_vectors[29],
    latent_vectors[33], latent_vectors[38], latent_vectors[39], latent_vectors[43], latent_vectors[155], latent_vectors[275],  latent_vectors[294],
    latent_vectors[388], latent_vectors[482], latent_vectors[202], latent_vectors[185], latent_vectors[76], latent_vectors[79], latent_vectors[60],
    latent_vectors[120], latent_vectors[115], latent_vectors[132], latent_vectors[135], latent_vectors[139], latent_vectors[142], latent_vectors[150],
    latent_vectors[218], latent_vectors[229], latent_vectors[234], latent_vectors[299], latent_vectors[308], latent_vectors[453], latent_vectors[467],
    latent_vectors[489], latent_vectors[613], latent_vectors[614], latent_vectors[672], latent_vectors[703], latent_vectors[753], latent_vectors[801],
    latent_vectors[521], latent_vectors[621], latent_vectors[669], latent_vectors[693], latent_vectors[736], latent_vectors[829], latent_vectors[862],
    latent_vectors[739], latent_vectors[779], latent_vectors[809], latent_vectors[888], latent_vectors[889], latent_vectors[890], latent_vectors[904],
    latent_vectors[911], latent_vectors[917], latent_vectors[919], latent_vectors[972]   ]

    with_bangs_hair = [latent_vectors[8], latent_vectors[12], latent_vectors[13], latent_vectors[15], latent_vectors[16], latent_vectors[154], 
    latent_vectors[485], latent_vectors[389], latent_vectors[392], latent_vectors[499], latent_vectors[461], latent_vectors[417], latent_vectors[401], 
    latent_vectors[400], latent_vectors[371], latent_vectors[363], latent_vectors[358], latent_vectors[341], latent_vectors[298], latent_vectors[293], 
    latent_vectors[292], latent_vectors[282], latent_vectors[263], latent_vectors[255], latent_vectors[249], latent_vectors[236], latent_vectors[189], 
    latent_vectors[182], latent_vectors[178], latent_vectors[164], latent_vectors[153], latent_vectors[148], latent_vectors[148], latent_vectors[140],
    latent_vectors[505], latent_vectors[548], latent_vectors[565], latent_vectors[566], latent_vectors[593], latent_vectors[601], latent_vectors[611],
    latent_vectors[533], latent_vectors[536], latent_vectors[585], latent_vectors[612], latent_vectors[622], latent_vectors[628], latent_vectors[679],
    latent_vectors[577], latent_vectors[689], latent_vectors[695], latent_vectors[709], latent_vectors[728], latent_vectors[743], latent_vectors[746],
    latent_vectors[730], latent_vectors[733], latent_vectors[759], latent_vectors[761], latent_vectors[810], latent_vectors[820], latent_vectors[837],
    latent_vectors[838], latent_vectors[855], latent_vectors[857], latent_vectors[858], latent_vectors[866], latent_vectors[869], latent_vectors[924]  ]

    bald_hair       = [ latent_vectors[40], latent_vectors[51], latent_vectors[173], latent_vectors[267], latent_vectors[406], latent_vectors[486],
    latent_vectors[630], latent_vectors[796], latent_vectors[806], latent_vectors[851], latent_vectors[871]   ]




    # holds the latent vectors for every feature in hair length
    short_hair = [latent_vectors[0], latent_vectors[2], latent_vectors[5], latent_vectors[6], latent_vectors[9], latent_vectors[13], latent_vectors[14],
    latent_vectors[15], latent_vectors[16], latent_vectors[17], latent_vectors[18], latent_vectors[21], latent_vectors[23], latent_vectors[24],
    latent_vectors[25], latent_vectors[26], latent_vectors[28], latent_vectors[31], latent_vectors[33], latent_vectors[35],  latent_vectors[37],
    latent_vectors[38], latent_vectors[39], latent_vectors[40], latent_vectors[43], latent_vectors[156], latent_vectors[163], latent_vectors[169],
    latent_vectors[186], latent_vectors[190], latent_vectors[193], latent_vectors[281], latent_vectors[279], latent_vectors[275], latent_vectors[273],
    latent_vectors[1000], latent_vectors[997], latent_vectors[980], latent_vectors[905], latent_vectors[892], latent_vectors[891], latent_vectors[881],
    latent_vectors[992], latent_vectors[983], latent_vectors[911], latent_vectors[900], latent_vectors[864], latent_vectors[842], latent_vectors[784],
    latent_vectors[954], latent_vectors[940], latent_vectors[914], latent_vectors[899], latent_vectors[886], latent_vectors[804], latent_vectors[793],
    latent_vectors[977], latent_vectors[958], latent_vectors[952], latent_vectors[897], latent_vectors[866], latent_vectors[791], latent_vectors[782],
    latent_vectors[935], latent_vectors[927], latent_vectors[919], latent_vectors[838], latent_vectors[801], latent_vectors[759], latent_vectors[743],
    latent_vectors[730], latent_vectors[728], latent_vectors[723], latent_vectors[720], latent_vectors[709], latent_vectors[684], latent_vectors[878]  ]

    long_hair  = [latent_vectors[1], latent_vectors[3], latent_vectors[4], latent_vectors[7], latent_vectors[8], latent_vectors[10], latent_vectors[11],
    latent_vectors[12], latent_vectors[19],  latent_vectors[20], latent_vectors[22], latent_vectors[27], latent_vectors[29], latent_vectors[32],
    latent_vectors[34], latent_vectors[36], latent_vectors[41], latent_vectors[42], latent_vectors[44],latent_vectors[152], latent_vectors[160],
    latent_vectors[161], latent_vectors[162], latent_vectors[166], latent_vectors[174], latent_vectors[189], latent_vectors[195], latent_vectors[280],
    latent_vectors[295], latent_vectors[433], latent_vectors[496], latent_vectors[485], latent_vectors[478], latent_vectors[473], latent_vectors[476],
    latent_vectors[995], latent_vectors[988], latent_vectors[978], latent_vectors[947], latent_vectors[834], latent_vectors[802], latent_vectors[789],
    latent_vectors[975], latent_vectors[966], latent_vectors[936], latent_vectors[857], latent_vectors[849], latent_vectors[783], latent_vectors[748],
    latent_vectors[956], latent_vectors[928], latent_vectors[901], latent_vectors[855], latent_vectors[837], latent_vectors[761], latent_vectors[694],
    latent_vectors[970], latent_vectors[968], latent_vectors[878], latent_vectors[800], latent_vectors[786], latent_vectors[735], latent_vectors[690],
    latent_vectors[931], latent_vectors[924], latent_vectors[896], latent_vectors[798], latent_vectors[785], latent_vectors[683], latent_vectors[677],
    latent_vectors[733], latent_vectors[727], latent_vectors[724], latent_vectors[686], latent_vectors[643], latent_vectors[640], latent_vectors[577] ]




    # holds the latent vectors for every feature in eye color
    black_eye = [latent_vectors[0], latent_vectors[6], latent_vectors[14], latent_vectors[16], latent_vectors[24], latent_vectors[30], latent_vectors[37],
    latent_vectors[48], latent_vectors[49], latent_vectors[61], latent_vectors[63], latent_vectors[68], latent_vectors[83], latent_vectors[86],
    latent_vectors[118], latent_vectors[151], latent_vectors[165], latent_vectors[167], latent_vectors[168], latent_vectors[173], latent_vectors[185],
    latent_vectors[219], latent_vectors[237], latent_vectors[247], latent_vectors[261], latent_vectors[262], latent_vectors[263], latent_vectors[291],
    latent_vectors[300], latent_vectors[303], latent_vectors[305], latent_vectors[313], latent_vectors[336], latent_vectors[381], latent_vectors[399],
    latent_vectors[504], latent_vectors[506], latent_vectors[559], latent_vectors[659], latent_vectors[687], latent_vectors[812], latent_vectors[933],
    latent_vectors[512], latent_vectors[523], latent_vectors[550], latent_vectors[690], latent_vectors[857], latent_vectors[880], latent_vectors[935],
    latent_vectors[540], latent_vectors[543], latent_vectors[639], latent_vectors[734], latent_vectors[968], latent_vectors[972], latent_vectors[995],
    latent_vectors[701], latent_vectors[708], latent_vectors[717], latent_vectors[773], latent_vectors[996]   ]

    blue_eye  = [latent_vectors[8], latent_vectors[10], latent_vectors[15], latent_vectors[20], latent_vectors[22], latent_vectors[26], latent_vectors[28],
    latent_vectors[31], latent_vectors[33], latent_vectors[43], latent_vectors[50], latent_vectors[52], latent_vectors[70], latent_vectors[77],
    latent_vectors[79], latent_vectors[95], latent_vectors[109], latent_vectors[111], latent_vectors[132], latent_vectors[139], latent_vectors[146],
    latent_vectors[149], latent_vectors[166], latent_vectors[186], latent_vectors[189], latent_vectors[190], latent_vectors[192], latent_vectors[198],
    latent_vectors[205], latent_vectors[225], latent_vectors[229], latent_vectors[241], latent_vectors[249], latent_vectors[256], latent_vectors[257],
    latent_vectors[517], latent_vectors[531], latent_vectors[533], latent_vectors[807], latent_vectors[869], latent_vectors[871], latent_vectors[886],
    latent_vectors[541], latent_vectors[587], latent_vectors[605], latent_vectors[892], latent_vectors[911], latent_vectors[917], latent_vectors[924],
    latent_vectors[631], latent_vectors[733], latent_vectors[774], latent_vectors[934], latent_vectors[948], latent_vectors[957], latent_vectors[994],
    latent_vectors[835], latent_vectors[842], latent_vectors[878]  ]

    green_eye = [latent_vectors[1], latent_vectors[4], latent_vectors[5], latent_vectors[12], latent_vectors[13], latent_vectors[17], latent_vectors[19],
    latent_vectors[25], latent_vectors[32], latent_vectors[35], latent_vectors[38], latent_vectors[40], latent_vectors[42], latent_vectors[54],
    latent_vectors[73], latent_vectors[75], latent_vectors[101], latent_vectors[106], latent_vectors[113],latent_vectors[114], latent_vectors[136],
    latent_vectors[152], latent_vectors[153],latent_vectors[162], latent_vectors[216], latent_vectors[223], latent_vectors[239], latent_vectors[266],
    latent_vectors[275], latent_vectors[286], latent_vectors[288],  latent_vectors[293], latent_vectors[319], latent_vectors[335], latent_vectors[346],
    latent_vectors[503], latent_vectors[508], latent_vectors[513], latent_vectors[655], latent_vectors[706], latent_vectors[745], latent_vectors[747],
    latent_vectors[522], latent_vectors[600], latent_vectors[628], latent_vectors[699], latent_vectors[718], latent_vectors[763], latent_vectors[765],
    latent_vectors[560], latent_vectors[561], latent_vectors[637], latent_vectors[786], latent_vectors[794], latent_vectors[801], latent_vectors[824],
    latent_vectors[574], latent_vectors[597], latent_vectors[695], latent_vectors[849], latent_vectors[895], latent_vectors[899], latent_vectors[956]  ]

    brown_eye = [latent_vectors[2], latent_vectors[3], latent_vectors[7], latent_vectors[9], latent_vectors[11], latent_vectors[18], latent_vectors[21],
    latent_vectors[23], latent_vectors[29], latent_vectors[36], latent_vectors[39], latent_vectors[41], latent_vectors[44], latent_vectors[71],
    latent_vectors[74], latent_vectors[124], latent_vectors[131], latent_vectors[159], latent_vectors[160], latent_vectors[195], latent_vectors[208],
    latent_vectors[217], latent_vectors[268], latent_vectors[269], latent_vectors[279], latent_vectors[284], latent_vectors[297], latent_vectors[307],
    latent_vectors[316], latent_vectors[321], latent_vectors[380], latent_vectors[387], latent_vectors[402], latent_vectors[410], latent_vectors[422],
    latent_vectors[516], latent_vectors[528], latent_vectors[535], latent_vectors[663], latent_vectors[668], latent_vectors[671], latent_vectors[682],
    latent_vectors[538], latent_vectors[546], latent_vectors[585], latent_vectors[686], latent_vectors[749], latent_vectors[755], latent_vectors[843],
    latent_vectors[545], latent_vectors[579], latent_vectors[610], latent_vectors[760], latent_vectors[783], latent_vectors[795], latent_vectors[866],
    latent_vectors[588], latent_vectors[592], latent_vectors[643], latent_vectors[913], latent_vectors[915], latent_vectors[932], latent_vectors[947]  ]



    # holds the latent vectors for every feature in bag under eye
    on_bags_eye  = [ latent_vectors[1], latent_vectors[76], latent_vectors[997], latent_vectors[976], latent_vectors[781], latent_vectors[721],
    latent_vectors[934], latent_vectors[877], latent_vectors[849], latent_vectors[823], latent_vectors[625], latent_vectors[589], latent_vectors[575],
    latent_vectors[921], latent_vectors[913], latent_vectors[848], latent_vectors[766], latent_vectors[622], latent_vectors[574], latent_vectors[401],
    latent_vectors[900], latent_vectors[870], latent_vectors[835], latent_vectors[761], latent_vectors[614], latent_vectors[462], latent_vectors[399],
    latent_vectors[895], latent_vectors[866], latent_vectors[839], latent_vectors[710], latent_vectors[611], latent_vectors[383], latent_vectors[378],
    latent_vectors[707], latent_vectors[695], latent_vectors[690], latent_vectors[635], latent_vectors[610], latent_vectors[362], latent_vectors[338],
    latent_vectors[630], latent_vectors[571]   ]

    off_bags_eye = [ latent_vectors[0], latent_vectors[2], latent_vectors[3], latent_vectors[6], latent_vectors[7], latent_vectors[8], latent_vectors[10],
    latent_vectors[11], latent_vectors[12], latent_vectors[13], latent_vectors[14], latent_vectors[15], latent_vectors[16], latent_vectors[17], 
    latent_vectors[18], latent_vectors[19], latent_vectors[20], latent_vectors[21], latent_vectors[22], latent_vectors[23], latent_vectors[25],
    latent_vectors[28], latent_vectors[31], latent_vectors[32], latent_vectors[35], latent_vectors[36], latent_vectors[39], latent_vectors[41],
    latent_vectors[43], latent_vectors[44], latent_vectors[470], latent_vectors[356], latent_vectors[350], latent_vectors[346], latent_vectors[341],
    latent_vectors[985], latent_vectors[983], latent_vectors[982], latent_vectors[753], latent_vectors[648], latent_vectors[980], latent_vectors[956], 
    latent_vectors[743], latent_vectors[978], latent_vectors[953], latent_vectors[975], latent_vectors[942], latent_vectors[560], latent_vectors[538],
    latent_vectors[957], latent_vectors[602]   ]




    # holds the latent vectors for every feature in facial hair
    no_beard = [ latent_vectors[0], latent_vectors[1], latent_vectors[315], latent_vectors[117], latent_vectors[114], latent_vectors[119], latent_vectors[108],
    latent_vectors[107], latent_vectors[106], latent_vectors[103], latent_vectors[100], latent_vectors[99], latent_vectors[96], latent_vectors[95],
    latent_vectors[93], latent_vectors[91], latent_vectors[90], latent_vectors[87], latent_vectors[83], latent_vectors[82], latent_vectors[81],
    latent_vectors[79], latent_vectors[74], latent_vectors[68], latent_vectors[65], latent_vectors[62], latent_vectors[61], latent_vectors[60],
    latent_vectors[58], latent_vectors[57], latent_vectors[55], latent_vectors[50], latent_vectors[45], latent_vectors[42], latent_vectors[40],
    latent_vectors[699], latent_vectors[703], latent_vectors[705], latent_vectors[707], latent_vectors[709], latent_vectors[711], latent_vectors[719],
    latent_vectors[724], latent_vectors[727], latent_vectors[728], latent_vectors[729], latent_vectors[741], latent_vectors[742], latent_vectors[743],
    latent_vectors[720], latent_vectors[745], latent_vectors[747], latent_vectors[758], latent_vectors[759], latent_vectors[761], latent_vectors[765],
    latent_vectors[723], latent_vectors[755], latent_vectors[757], latent_vectors[767], latent_vectors[771], latent_vectors[785], latent_vectors[792],
    latent_vectors[756], latent_vectors[772], latent_vectors[783], latent_vectors[790], latent_vectors[791], latent_vectors[793], latent_vectors[798]  ]
    
    goatees = [ latent_vectors[2], latent_vectors[24], latent_vectors[123], latent_vectors[191], latent_vectors[222], latent_vectors[284], latent_vectors[331],
    latent_vectors[339], latent_vectors[374], latent_vectors[418], latent_vectors[489], latent_vectors[967], latent_vectors[962], latent_vectors[753],
    latent_vectors[722], latent_vectors[669], latent_vectors[637], latent_vectors[567], latent_vectors[543]  ]

    mustaches= [ latent_vectors[2], latent_vectors[24], latent_vectors[30], latent_vectors[38], latent_vectors[51], latent_vectors[76], latent_vectors[84],
    latent_vectors[88], latent_vectors[123], latent_vectors[128], latent_vectors[134], latent_vectors[172], latent_vectors[186], latent_vectors[208],
    latent_vectors[227], latent_vectors[248], latent_vectors[253], latent_vectors[267], latent_vectors[272], latent_vectors[275], latent_vectors[296],
    latent_vectors[301], latent_vectors[302], latent_vectors[329], latent_vectors[331], latent_vectors[370], latent_vectors[378], latent_vectors[384],
    latent_vectors[430], latent_vectors[475], latent_vectors[495], latent_vectors[954], latent_vectors[897], latent_vectors[766], latent_vectors[763],
    latent_vectors[748], latent_vectors[693], latent_vectors[669], latent_vectors[637], latent_vectors[629], latent_vectors[624], latent_vectors[623],
    latent_vectors[623], latent_vectors[614], latent_vectors[578], latent_vectors[567], latent_vectors[522], latent_vectors[521], latent_vectors[503]  ]

    beards = [ latent_vectors[30], latent_vectors[38], latent_vectors[76], latent_vectors[84], latent_vectors[88], latent_vectors[139], latent_vectors[186],
    latent_vectors[208], latent_vectors[227], latent_vectors[248], latent_vectors[252], latent_vectors[253], latent_vectors[265], latent_vectors[267],
    latent_vectors[271], latent_vectors[272], latent_vectors[291], latent_vectors[301], latent_vectors[302], latent_vectors[329], latent_vectors[367],
    latent_vectors[378], latent_vectors[384], latent_vectors[403], latent_vectors[405], latent_vectors[452], latent_vectors[475], latent_vectors[482],
    latent_vectors[487], latent_vectors[495], latent_vectors[954], latent_vectors[945], latent_vectors[921], latent_vectors[909], latent_vectors[897], 
    latent_vectors[894], latent_vectors[874], latent_vectors[868], latent_vectors[856], latent_vectors[796], latent_vectors[782], latent_vectors[779],
    latent_vectors[774], latent_vectors[489], latent_vectors[763], latent_vectors[748], latent_vectors[736], latent_vectors[718], latent_vectors[693],
    latent_vectors[629], latent_vectors[623], latent_vectors[614], latent_vectors[602], latent_vectors[578], latent_vectors[572], latent_vectors[570],
    latent_vectors[532], latent_vectors[522], latent_vectors[521], latent_vectors[503] ]


    # holds the latent vectors for every feature in cheeks
    normal_cheeks = [latent_vectors[1], latent_vectors[2], latent_vectors[6], latent_vectors[52], latent_vectors[58], latent_vectors[60], latent_vectors[86],
    latent_vectors[105], latent_vectors[12], latent_vectors[14], latent_vectors[21], latent_vectors[25], latent_vectors[36], latent_vectors[43],
    latent_vectors[48], latent_vectors[55], latent_vectors[61], latent_vectors[67], latent_vectors[75], latent_vectors[77], latent_vectors[91],
    latent_vectors[103], latent_vectors[105], latent_vectors[108], latent_vectors[114], latent_vectors[117], latent_vectors[120], latent_vectors[121],
    latent_vectors[124], latent_vectors[132], latent_vectors[139], latent_vectors[147], latent_vectors[151], latent_vectors[162], latent_vectors[185],
    latent_vectors[504], latent_vectors[510], latent_vectors[511], latent_vectors[512], latent_vectors[514], latent_vectors[531], latent_vectors[532],
    latent_vectors[534], latent_vectors[538], latent_vectors[540], latent_vectors[543], latent_vectors[547], latent_vectors[559], latent_vectors[564],
    latent_vectors[565], latent_vectors[567], latent_vectors[568], latent_vectors[569], latent_vectors[570], latent_vectors[575], latent_vectors[583],
    latent_vectors[587], latent_vectors[589], latent_vectors[593], latent_vectors[605], latent_vectors[614], latent_vectors[615], latent_vectors[625],
    latent_vectors[629], latent_vectors[630], latent_vectors[631], latent_vectors[633], latent_vectors[642], latent_vectors[653], latent_vectors[678]  ]

    rosy_cheeks   = [latent_vectors[13], latent_vectors[28], latent_vectors[8], latent_vectors[22], latent_vectors[50], latent_vectors[56], latent_vectors[57],
    latent_vectors[135], latent_vectors[63], latent_vectors[65], latent_vectors[70], latent_vectors[80], latent_vectors[911], latent_vectors[112],
    latent_vectors[113], latent_vectors[123], latent_vectors[146], latent_vectors[169], latent_vectors[180], latent_vectors[197], latent_vectors[205],
    latent_vectors[208], latent_vectors[233], latent_vectors[240], latent_vectors[241], latent_vectors[250], latent_vectors[258], latent_vectors[265],
    latent_vectors[274], latent_vectors[278], latent_vectors[312], latent_vectors[328], latent_vectors[345], latent_vectors[351], latent_vectors[367],
    latent_vectors[506], latent_vectors[518], latent_vectors[522], latent_vectors[536], latent_vectors[545], latent_vectors[562], latent_vectors[572],
    latent_vectors[576], latent_vectors[586], latent_vectors[590], latent_vectors[592], latent_vectors[595], latent_vectors[602], latent_vectors[603],
    latent_vectors[610], latent_vectors[650], latent_vectors[651], latent_vectors[652], latent_vectors[654], latent_vectors[669], latent_vectors[698],
    latent_vectors[959], latent_vectors[969], latent_vectors[977], latent_vectors[980], latent_vectors[986], latent_vectors[998],   ]



    # holds the latent vectors for every feature in face shape
    oval_face = [latent_vectors[4], latent_vectors[6], latent_vectors[15], latent_vectors[25], latent_vectors[28], latent_vectors[42], latent_vectors[52],
    latent_vectors[61], latent_vectors[70], latent_vectors[104], latent_vectors[106], latent_vectors[113], latent_vectors[122], latent_vectors[134],
    latent_vectors[139], latent_vectors[146], latent_vectors[149], latent_vectors[155], latent_vectors[157], latent_vectors[169], latent_vectors[225],
    latent_vectors[236], latent_vectors[247], latent_vectors[252], latent_vectors[258], latent_vectors[278], latent_vectors[282], latent_vectors[296],
    latent_vectors[299], latent_vectors[302], latent_vectors[306], latent_vectors[317], latent_vectors[321], latent_vectors[331], latent_vectors[341],
    latent_vectors[800], latent_vectors[803], latent_vectors[807], latent_vectors[821], latent_vectors[825], latent_vectors[828], latent_vectors[832], 
    latent_vectors[835], latent_vectors[839], latent_vectors[848], latent_vectors[856], latent_vectors[867], latent_vectors[868], latent_vectors[872], 
    latent_vectors[875], latent_vectors[877], latent_vectors[885], latent_vectors[892], latent_vectors[906], latent_vectors[911], latent_vectors[923], 
    latent_vectors[939], latent_vectors[946], latent_vectors[952], latent_vectors[961], latent_vectors[967], latent_vectors[975], latent_vectors[997]  ]


    circular_face = [latent_vectors[0], latent_vectors[3], latent_vectors[16], latent_vectors[33], latent_vectors[58], latent_vectors[60], latent_vectors[63],
    latent_vectors[68], latent_vectors[105], latent_vectors[111], latent_vectors[114], latent_vectors[117], latent_vectors[120], latent_vectors[121],
    latent_vectors[129], latent_vectors[141], latent_vectors[156], latent_vectors[168], latent_vectors[170], latent_vectors[184], latent_vectors[233],
    latent_vectors[234], latent_vectors[257], latent_vectors[281], latent_vectors[289], latent_vectors[294], latent_vectors[298], latent_vectors[304],
    latent_vectors[308], latent_vectors[320], latent_vectors[323], latent_vectors[327], latent_vectors[334], latent_vectors[340], latent_vectors[356],
    latent_vectors[814], latent_vectors[818], latent_vectors[820], latent_vectors[822], latent_vectors[827], latent_vectors[837], latent_vectors[845], 
    latent_vectors[849], latent_vectors[851], latent_vectors[854], latent_vectors[857], latent_vectors[864], latent_vectors[865], latent_vectors[876], 
    latent_vectors[881], latent_vectors[882], latent_vectors[883], latent_vectors[886], latent_vectors[888], latent_vectors[890], latent_vectors[891], 
    latent_vectors[894], latent_vectors[900], latent_vectors[903], latent_vectors[915], latent_vectors[917], latent_vectors[927], latent_vectors[937]   ]


    # holds the latent vectors for every feature in ears
    small_ears = [latent_vectors[0], latent_vectors[54], latent_vectors[56], latent_vectors[57], latent_vectors[60], latent_vectors[63], latent_vectors[68],
    latent_vectors[73], latent_vectors[81], latent_vectors[84], latent_vectors[85], latent_vectors[86], latent_vectors[88], latent_vectors[91],
    latent_vectors[93], latent_vectors[97], latent_vectors[102], latent_vectors[106], latent_vectors[107], latent_vectors[115], latent_vectors[117],
    latent_vectors[120], latent_vectors[133], latent_vectors[154], latent_vectors[156], latent_vectors[169], latent_vectors[175], latent_vectors[183],
    latent_vectors[185], latent_vectors[186], latent_vectors[187], latent_vectors[193], latent_vectors[194], latent_vectors[211], latent_vectors[223],
    latent_vectors[997], latent_vectors[980], latent_vectors[974], latent_vectors[972], latent_vectors[961], latent_vectors[946], latent_vectors[933], 
    latent_vectors[916], latent_vectors[915], latent_vectors[909], latent_vectors[888], latent_vectors[880], latent_vectors[841], latent_vectors[821], 
    latent_vectors[908], latent_vectors[900], latent_vectors[865], latent_vectors[850], latent_vectors[837], latent_vectors[827], latent_vectors[808], 
    latent_vectors[854], latent_vectors[806], latent_vectors[792], latent_vectors[779], latent_vectors[766], latent_vectors[736], latent_vectors[727]   ]

    big_ears   = [latent_vectors[13], latent_vectors[55], latent_vectors[58], latent_vectors[79], latent_vectors[82], latent_vectors[87], latent_vectors[108],
    latent_vectors[123], latent_vectors[134], latent_vectors[141], latent_vectors[150], latent_vectors[153], latent_vectors[155], latent_vectors[173],
    latent_vectors[177], latent_vectors[180], latent_vectors[202], latent_vectors[225], latent_vectors[229], latent_vectors[257], latent_vectors[264],
    latent_vectors[273], latent_vectors[275], latent_vectors[289], latent_vectors[291], latent_vectors[302], latent_vectors[327], latent_vectors[266],
    latent_vectors[492], latent_vectors[493], latent_vectors[481], latent_vectors[453], latent_vectors[324], latent_vectors[297], latent_vectors[207],
    latent_vectors[952], latent_vectors[927], latent_vectors[919], latent_vectors[911], latent_vectors[906], latent_vectors[891], latent_vectors[848], 
    latent_vectors[897], latent_vectors[895], latent_vectors[871], latent_vectors[851], latent_vectors[811], latent_vectors[805], latent_vectors[559], 
    latent_vectors[894], latent_vectors[835], latent_vectors[822], latent_vectors[807], latent_vectors[681], latent_vectors[527], latent_vectors[501], 
    latent_vectors[872], latent_vectors[829], latent_vectors[753]   ]


    # holds the latent vectors for every feature in head wear
    headwear_off= [latent_vectors[0], latent_vectors[315], latent_vectors[314], latent_vectors[311], latent_vectors[310], latent_vectors[308],
    latent_vectors[307], latent_vectors[306], latent_vectors[302], latent_vectors[298], latent_vectors[299], latent_vectors[293], latent_vectors[291],
    latent_vectors[290], latent_vectors[289], latent_vectors[286], latent_vectors[284], latent_vectors[282], latent_vectors[280], latent_vectors[279],
    latent_vectors[278], latent_vectors[277], latent_vectors[273], latent_vectors[272], latent_vectors[271], latent_vectors[267], latent_vectors[266],
    latent_vectors[262], latent_vectors[257], latent_vectors[250], latent_vectors[245], latent_vectors[243], latent_vectors[242], latent_vectors[236],
    latent_vectors[500], latent_vectors[501], latent_vectors[503], latent_vectors[506], latent_vectors[508], latent_vectors[509], latent_vectors[510],
    latent_vectors[511], latent_vectors[520], latent_vectors[531], latent_vectors[549], latent_vectors[559], latent_vectors[571], latent_vectors[572],
    latent_vectors[521], latent_vectors[530], latent_vectors[546], latent_vectors[555], latent_vectors[564], latent_vectors[567], latent_vectors[477],
    latent_vectors[526], latent_vectors[532], latent_vectors[536], latent_vectors[540], latent_vectors[543], latent_vectors[562], latent_vectors[565]  ]

    headwear_on = [ latent_vectors[313], latent_vectors[312], latent_vectors[274], latent_vectors[264], latent_vectors[263], latent_vectors[241],
    latent_vectors[238], latent_vectors[231], latent_vectors[230], latent_vectors[227], latent_vectors[226], latent_vectors[220], latent_vectors[200],
    latent_vectors[199], latent_vectors[198], latent_vectors[192], latent_vectors[184], latent_vectors[167], latent_vectors[165], latent_vectors[157],
    latent_vectors[153], latent_vectors[136], latent_vectors[128], latent_vectors[116], latent_vectors[112], latent_vectors[82], latent_vectors[77],
    latent_vectors[43], latent_vectors[470], latent_vectors[468], latent_vectors[454], latent_vectors[431], latent_vectors[418], latent_vectors[334],
    latent_vectors[519], latent_vectors[560], latent_vectors[570], latent_vectors[661], latent_vectors[768], latent_vectors[770], latent_vectors[778],
    latent_vectors[552], latent_vectors[558], latent_vectors[573], latent_vectors[578], latent_vectors[825], latent_vectors[921], latent_vectors[937],
    latent_vectors[553], latent_vectors[663], latent_vectors[704], latent_vectors[800], latent_vectors[865], latent_vectors[870], latent_vectors[940],
    latent_vectors[594], latent_vectors[664], latent_vectors[580], latent_vectors[884], latent_vectors[967]   ]


    # holds the latent vectors for every feature in double chin
    double_chin_off = [ latent_vectors[93], latent_vectors[113], latent_vectors[328], latent_vectors[322], latent_vectors[321], latent_vectors[318],
    latent_vectors[317], latent_vectors[316], latent_vectors[314], latent_vectors[310], latent_vectors[306], latent_vectors[300], latent_vectors[291],
    latent_vectors[289], latent_vectors[288], latent_vectors[281], latent_vectors[271], latent_vectors[253], latent_vectors[264], latent_vectors[260],
    latent_vectors[251], latent_vectors[249], latent_vectors[248], latent_vectors[237], latent_vectors[233], latent_vectors[218], latent_vectors[217],
    latent_vectors[213], latent_vectors[211], latent_vectors[208], latent_vectors[200], latent_vectors[191], latent_vectors[183], latent_vectors[174],
    latent_vectors[985], latent_vectors[969], latent_vectors[961], latent_vectors[947], latent_vectors[914], latent_vectors[906], latent_vectors[896],
    latent_vectors[983], latent_vectors[921], latent_vectors[895], latent_vectors[842], latent_vectors[813], latent_vectors[748], latent_vectors[747],
    latent_vectors[967], latent_vectors[954], latent_vectors[892], latent_vectors[883], latent_vectors[827], latent_vectors[763], latent_vectors[758],
    latent_vectors[952], latent_vectors[857], latent_vectors[822], latent_vectors[820], latent_vectors[798], latent_vectors[786], latent_vectors[774]  ]

    double_chin_on = [latent_vectors[6], latent_vectors[87], latent_vectors[120], latent_vectors[163], latent_vectors[170], latent_vectors[173],
    latent_vectors[180], latent_vectors[261], latent_vectors[275], latent_vectors[278], latent_vectors[294], latent_vectors[299], latent_vectors[344],
    latent_vectors[332], latent_vectors[331], latent_vectors[312], latent_vectors[305], latent_vectors[286], latent_vectors[268], latent_vectors[254],
    latent_vectors[296], latent_vectors[273], latent_vectors[269], latent_vectors[243], latent_vectors[238], latent_vectors[229], latent_vectors[223],
    latent_vectors[220], latent_vectors[219], latent_vectors[203], latent_vectors[175], latent_vectors[122], latent_vectors[92], latent_vectors[80],
    latent_vectors[981], latent_vectors[974], latent_vectors[960], latent_vectors[915], latent_vectors[908], latent_vectors[730], latent_vectors[656],
    latent_vectors[977], latent_vectors[930], latent_vectors[903], latent_vectors[759], latent_vectors[658], latent_vectors[651], latent_vectors[632],
    latent_vectors[937], latent_vectors[912], latent_vectors[890], latent_vectors[756], latent_vectors[676], latent_vectors[644], latent_vectors[619],
    latent_vectors[923], latent_vectors[845], latent_vectors[812], latent_vectors[793], latent_vectors[741], latent_vectors[737], latent_vectors[703] ]


    # holds the latent vectors for every feature in lipstick
    lipstick_off= [ latent_vectors[1], latent_vectors[356], latent_vectors[355], latent_vectors[354], latent_vectors[341], latent_vectors[315], 
    latent_vectors[40], latent_vectors[34], latent_vectors[33], latent_vectors[29], latent_vectors[27], latent_vectors[25], latent_vectors[23],
    latent_vectors[22], latent_vectors[18], latent_vectors[18], latent_vectors[13], latent_vectors[3], latent_vectors[499], latent_vectors[498],
    latent_vectors[497], latent_vectors[494], latent_vectors[493], latent_vectors[491], latent_vectors[490], latent_vectors[488], latent_vectors[486],
    latent_vectors[482], latent_vectors[479], latent_vectors[478], latent_vectors[477], latent_vectors[471], latent_vectors[467], latent_vectors[463],
    latent_vectors[540], latent_vectors[534], latent_vectors[575], latent_vectors[614], latent_vectors[635], latent_vectors[641], latent_vectors[646],
    latent_vectors[506], latent_vectors[526], latent_vectors[558], latent_vectors[559], latent_vectors[560], latent_vectors[610], latent_vectors[653],
    latent_vectors[509], latent_vectors[522], latent_vectors[528], latent_vectors[566], latent_vectors[569], latent_vectors[630], latent_vectors[681],
    latent_vectors[523], latent_vectors[531], latent_vectors[563], latent_vectors[564], latent_vectors[607], latent_vectors[619], latent_vectors[625]  ]

    lipstick_on = [ latent_vectors[343], latent_vectors[20], latent_vectors[11], latent_vectors[4], latent_vectors[470], latent_vectors[469],
    latent_vectors[451], latent_vectors[450], latent_vectors[432], latent_vectors[408], latent_vectors[402], latent_vectors[379], latent_vectors[349],
    latent_vectors[347], latent_vectors[342], latent_vectors[323], latent_vectors[312], latent_vectors[305], latent_vectors[304], latent_vectors[295],
    latent_vectors[285], latent_vectors[260], latent_vectors[256], latent_vectors[251], latent_vectors[247], latent_vectors[245], latent_vectors[237],
    latent_vectors[219], latent_vectors[217], latent_vectors[203], latent_vectors[195], latent_vectors[176], latent_vectors[167], latent_vectors[152],
    latent_vectors[539], latent_vectors[514], latent_vectors[554], latent_vectors[555], latent_vectors[605], latent_vectors[612], latent_vectors[632],
    latent_vectors[502], latent_vectors[594], latent_vectors[658], latent_vectors[664], latent_vectors[677], latent_vectors[777], latent_vectors[838],
    latent_vectors[640], latent_vectors[680], latent_vectors[694], latent_vectors[709], latent_vectors[875], latent_vectors[893], latent_vectors[910],
    latent_vectors[692], latent_vectors[730], latent_vectors[776], latent_vectors[780], latent_vectors[947], latent_vectors[952], latent_vectors[984]  ]



    # holds the latent vectors for every feature in eyebrows
    bushy_eyebrows  = [  latent_vectors[999], latent_vectors[997], latent_vectors[990], latent_vectors[989], latent_vectors[845], latent_vectors[977], 
    latent_vectors[961], latent_vectors[839], latent_vectors[969], latent_vectors[947], latent_vectors[796], latent_vectors[967], latent_vectors[921], 
    latent_vectors[785], latent_vectors[953], latent_vectors[898], latent_vectors[764], latent_vectors[937], latent_vectors[853], latent_vectors[918], 
    latent_vectors[847], latent_vectors[817], latent_vectors[604], latent_vectors[543], latent_vectors[534], latent_vectors[336], latent_vectors[324],
    latent_vectors[906], latent_vectors[880], latent_vectors[512], latent_vectors[194], latent_vectors[188], latent_vectors[147], latent_vectors[123]  ]


    arched_eyebrows = [  latent_vectors[995], latent_vectors[988], latent_vectors[877], latent_vectors[846], latent_vectors[987], latent_vectors[980], 
    latent_vectors[970], latent_vectors[854], latent_vectors[843], latent_vectors[985], latent_vectors[978], latent_vectors[931], latent_vectors[798], 
    latent_vectors[786], latent_vectors[983], latent_vectors[982], latent_vectors[927], latent_vectors[780], latent_vectors[976], latent_vectors[974], 
    latent_vectors[915], latent_vectors[763], latent_vectors[966], latent_vectors[952], latent_vectors[919], latent_vectors[765], latent_vectors[958], 
    latent_vectors[945], latent_vectors[914], latent_vectors[630], latent_vectors[956], latent_vectors[942], latent_vectors[899], latent_vectors[562],
    latent_vectors[950], latent_vectors[936], latent_vectors[892], latent_vectors[589], latent_vectors[522], latent_vectors[483], latent_vectors[409] ]




    # holds the latent vectors for every feature in makeup
    no_makeup    = [latent_vectors[1], latent_vectors[354], latent_vectors[336], latent_vectors[334], latent_vectors[332], latent_vectors[328], 
    latent_vectors[318], latent_vectors[316], latent_vectors[315], latent_vectors[180], latent_vectors[178], latent_vectors[174], latent_vectors[170],
    latent_vectors[168], latent_vectors[158], latent_vectors[156], latent_vectors[154], latent_vectors[153], latent_vectors[149], latent_vectors[146],
    latent_vectors[143], latent_vectors[141], latent_vectors[139], latent_vectors[138], latent_vectors[137], latent_vectors[136], latent_vectors[135],
    latent_vectors[134], latent_vectors[132], latent_vectors[126], latent_vectors[123], latent_vectors[122], latent_vectors[119], latent_vectors[117],
    latent_vectors[785], latent_vectors[760], latent_vectors[743], latent_vectors[654], latent_vectors[610], latent_vectors[577], latent_vectors[567],
    latent_vectors[782], latent_vectors[753], latent_vectors[540], latent_vectors[681], latent_vectors[527], latent_vectors[516], latent_vectors[506] ]

    # holds the latent vectors for every feature in makeup
    with_makeup = [latent_vectors[355], latent_vectors[36], latent_vectors[341], latent_vectors[335], latent_vectors[878], latent_vectors[875],
    latent_vectors[985], latent_vectors[970], latent_vectors[952], latent_vectors[826], latent_vectors[799], latent_vectors[797], latent_vectors[795],
    latent_vectors[924], latent_vectors[873], latent_vectors[857], latent_vectors[770], latent_vectors[709], latent_vectors[697], latent_vectors[692],
    latent_vectors[915], latent_vectors[546], latent_vectors[784], latent_vectors[730], latent_vectors[690], latent_vectors[685], latent_vectors[664],
    latent_vectors[910], latent_vectors[838], latent_vectors[780], latent_vectors[759], latent_vectors[677], latent_vectors[658], latent_vectors[640],
    latent_vectors[750], latent_vectors[747], latent_vectors[716], latent_vectors[643], latent_vectors[612], latent_vectors[605], latent_vectors[594],
    latent_vectors[582], latent_vectors[554], latent_vectors[549], latent_vectors[535], latent_vectors[514], latent_vectors[74], latent_vectors[476]  ]

    # holds the latent vectors for every feature in checkbones
    low_checkbones = [ latent_vectors[4], latent_vectors[67], latent_vectors[131], latent_vectors[141], latent_vectors[143], latent_vectors[149],
    latent_vectors[17], latent_vectors[24], latent_vectors[76], latent_vectors[124], latent_vectors[159], latent_vectors[162], latent_vectors[168],
    latent_vectors[21], latent_vectors[38], latent_vectors[89], latent_vectors[130], latent_vectors[151], latent_vectors[153], latent_vectors[178],
    latent_vectors[22], latent_vectors[51], latent_vectors[170], latent_vectors[203], latent_vectors[237], latent_vectors[310], latent_vectors[341]    ]

    high_checkbones= [ latent_vectors[0], latent_vectors[3], latent_vectors[81], latent_vectors[100], latent_vectors[188], latent_vectors[198],
    latent_vectors[13], latent_vectors[61], latent_vectors[68], latent_vectors[140], latent_vectors[156], latent_vectors[478], latent_vectors[479],
    latent_vectors[36], latent_vectors[63], latent_vectors[74], latent_vectors[199], latent_vectors[306], latent_vectors[400], latent_vectors[488],
    latent_vectors[317], latent_vectors[336], latent_vectors[549], latent_vectors[577], latent_vectors[582], latent_vectors[635], latent_vectors[590]    ]



# ------------------------------------------------------------------------------------------------------------------------------------------------------------

    # save the feature to each coresponding class
    Eye_Color[0] = black_eye
    Eye_Color[1] = blue_eye
    Eye_Color[2] = green_eye
    Eye_Color[3] = brown_eye

    # save the feature to each coresponding class
    Bags_Under_Eye[0] = off_bags_eye
    Bags_Under_Eye[1] = on_bags_eye

    # save the feature to each coresponding class
    Hair_Color[0] = black_hair
    Hair_Color[1] = blonde_hair
    Hair_Color[2] = brown_hair
    Hair_Color[3] = gray_hair

    # save the feature to each coresponding class
    Hair_Length[0] = short_hair
    Hair_Length[1] = long_hair

    # save the feature to each coresponding class
    Hair_style[0] = straight_hair
    Hair_style[1] = curly_hair
    Hair_style[2] = rec_hair
    Hair_style[3] = bald_hair
    Hair_style[4] = with_bangs_hair

    # save the feature to each coresponding class
    Cheeks[0] = normal_cheeks
    Cheeks[1] = rosy_cheeks

    # save the feature to each coresponding class
    Face_Shape[0] = oval_face
    Face_Shape[1] = circular_face

    # save the feature to each coresponding class
    Eyebrows[0] = arched_eyebrows
    Eyebrows[1] = bushy_eyebrows

    # save the feature to each coresponding class
    Ears[0] = small_ears
    Ears[1] = big_ears

    # save the feature to each coresponding class
    Cheekbones[0] = low_checkbones
    Cheekbones[1] = high_checkbones

    # save the feature to each coresponding class
    Double_Chin[0] = double_chin_off
    Double_Chin[1] = double_chin_on

    # save the feature to each coresponding class
    Facial_Hair[0] = no_beard
    Facial_Hair[1] = goatees
    Facial_Hair[2] = mustaches
    Facial_Hair[3] = beards
    #Facial_Hair[4] = sideburns

    # save the feature to each coresponding class
    Headwear[0] = headwear_off
    Headwear[1] = headwear_on

    # save the feature to each coresponding class
    Makeup[0] = no_makeup
    Makeup[1] = with_makeup

    # save the feature to each coresponding class
    Lipstick[0] = lipstick_off
    Lipstick[1] = lipstick_on

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ 

    # save the Eye color dataset
    with open('EyeColor.pickle', 'wb') as handle:
        pickle.dump(Eye_Color, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Bags_Under_Eye dataset
    with open('BagsUnderEye.pickle', 'wb') as handle:
        pickle.dump(Bags_Under_Eye, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Hair color dataset
    with open('HairColor.pickle', 'wb') as handle:
        pickle.dump(Hair_Color, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Hair Length dataset
    with open('HairLength.pickle', 'wb') as handle:
        pickle.dump(Hair_Length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the hair style dataset
    with open('HairStyle.pickle', 'wb') as handle:
        pickle.dump(Hair_style, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the cheeks dataset
    with open('Cheeks.pickle', 'wb') as handle:
        pickle.dump(Cheeks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the face shape dataset
    with open('FaceShape.pickle', 'wb') as handle:
        pickle.dump(Face_Shape, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Eyebrows dataset
    with open('EyeBrows.pickle', 'wb') as handle:
        pickle.dump(Eyebrows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the ears dataset
    with open('Ears.pickle', 'wb') as handle:
        pickle.dump(Ears, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the checkbones dataset
    with open('CheekBones.pickle', 'wb') as handle:
        pickle.dump(Cheekbones, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Double Chins dataset
    with open('DoubleChin.pickle', 'wb') as handle:
        pickle.dump(Double_Chin, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the facial hair dataset
    with open('FacialHair.pickle', 'wb') as handle:
        pickle.dump(Facial_Hair, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Head wear dataset
    with open('HeadWear.pickle', 'wb') as handle:
        pickle.dump(Headwear, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the Make up dataset
    with open('MakeUp.pickle', 'wb') as handle:
        pickle.dump(Makeup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the lipstick dataset
    with open('LipStick.pickle', 'wb') as handle:
        pickle.dump(Lipstick, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save()