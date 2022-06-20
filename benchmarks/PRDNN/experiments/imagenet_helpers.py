"""Helper methods for reading ImageNet-like datasets."""
import os
import pathlib
import random
from PIL import Image
import numpy as np

def read_imagenet_images(parent, n_labels=None, use_labels=None):
    """Reads a particular subset of the ImageNet dataset stored in @parent.

    The idea is that @parent has a number of subdirectories, each corresponding
    to a class, and under which there are a number of images. @n_labels gives
    the total number of labels you want. @use_labels limits it to a particular
    subset of the labels.

    Returns a pair (images, labels)

    Used by imagenet_patching.py
    """
    subdirs = [subdir for subdir in sorted(os.listdir(parent))
               if subdir != "README.txt"]
    if use_labels is not None:
        use_labels = list(map(SYNSETS.__getitem__, use_labels))
        subdirs = [subdir for subdir in subdirs
                   if int(subdir[1:]) in use_labels]
    if n_labels is not None:
        subdirs = subdirs[:n_labels]

    all_images, all_labels = [], []
    for subdir in subdirs:
        if subdir == "README.txt":
            continue
        synset_id = int(subdir[1:])
        label = SYNSETS.index(synset_id)
        images = sorted(os.listdir(f"{parent}/{subdir}"))
        random.shuffle(images)
        n_read = 0
        for image in images:
            path = pathlib.Path(f"{parent}/{subdir}/{image}")
            image = np.asarray(
                Image.open(path.resolve())
                .resize((224, 224))) / 255.
            if len(image.shape) == 3:
                all_images.append(image)
                all_labels.append(label)
                n_read += 1
            else:
                # Bad shape; we can de-BW it if we want.
                pass
    return np.array(all_images), np.array(all_labels)

# https://gist.github.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372
SYNSETS = list([
    1440764, 1443537, 1484850, 1491361, 1494475, 1496331, 1498041, 1514668,
    1514859, 1518878, 1530575, 1531178, 1532829, 1534433, 1537544, 1558993,
    1560419, 1580077, 1582220, 1592084, 1601694, 1608432, 1614925, 1616318,
    1622779, 1629819, 1630670, 1631663, 1632458, 1632777, 1641577, 1644373,
    1644900, 1664065, 1665541, 1667114, 1667778, 1669191, 1675722, 1677366,
    1682714, 1685808, 1687978, 1688243, 1689811, 1692333, 1693334, 1694178,
    1695060, 1697457, 1698640, 1704323, 1728572, 1728920, 1729322, 1729977,
    1734418, 1735189, 1737021, 1739381, 1740131, 1742172, 1744401, 1748264,
    1749939, 1751748, 1753488, 1755581, 1756291, 1768244, 1770081, 1770393,
    1773157, 1773549, 1773797, 1774384, 1774750, 1775062, 1776313, 1784675,
    1795545, 1796340, 1797886, 1798484, 1806143, 1806567, 1807496, 1817953,
    1818515, 1819313, 1820546, 1824575, 1828970, 1829413, 1833805, 1843065,
    1843383, 1847000, 1855032, 1855672, 1860187, 1871265, 1872401, 1873310,
    1877812, 1882714, 1883070, 1910747, 1914609, 1917289, 1924916, 1930112,
    1943899, 1944390, 1945685, 1950731, 1955084, 1968897, 1978287, 1978455,
    1980166, 1981276, 1983481, 1984695, 1985128, 1986214, 1990800, 2002556,
    2002724, 2006656, 2007558, 2009229, 2009912, 2011460, 2012849, 2013706,
    2017213, 2018207, 2018795, 2025239, 2027492, 2028035, 2033041, 2037110,
    2051845, 2056570, 2058221, 2066245, 2071294, 2074367, 2077923, 2085620,
    2085782, 2085936, 2086079, 2086240, 2086646, 2086910, 2087046, 2087394,
    2088094, 2088238, 2088364, 2088466, 2088632, 2089078, 2089867, 2089973,
    2090379, 2090622, 2090721, 2091032, 2091134, 2091244, 2091467, 2091635,
    2091831, 2092002, 2092339, 2093256, 2093428, 2093647, 2093754, 2093859,
    2093991, 2094114, 2094258, 2094433, 2095314, 2095570, 2095889, 2096051,
    2096177, 2096294, 2096437, 2096585, 2097047, 2097130, 2097209, 2097298,
    2097474, 2097658, 2098105, 2098286, 2098413, 2099267, 2099429, 2099601,
    2099712, 2099849, 2100236, 2100583, 2100735, 2100877, 2101006, 2101388,
    2101556, 2102040, 2102177, 2102318, 2102480, 2102973, 2104029, 2104365,
    2105056, 2105162, 2105251, 2105412, 2105505, 2105641, 2105855, 2106030,
    2106166, 2106382, 2106550, 2106662, 2107142, 2107312, 2107574, 2107683,
    2107908, 2108000, 2108089, 2108422, 2108551, 2108915, 2109047, 2109525,
    2109961, 2110063, 2110185, 2110341, 2110627, 2110806, 2110958, 2111129,
    2111277, 2111500, 2111889, 2112018, 2112137, 2112350, 2112706, 2113023,
    2113186, 2113624, 2113712, 2113799, 2113978, 2114367, 2114548, 2114712,
    2114855, 2115641, 2115913, 2116738, 2117135, 2119022, 2119789, 2120079,
    2120505, 2123045, 2123159, 2123394, 2123597, 2124075, 2125311, 2127052,
    2128385, 2128757, 2128925, 2129165, 2129604, 2130308, 2132136, 2133161,
    2134084, 2134418, 2137549, 2138441, 2165105, 2165456, 2167151, 2168699,
    2169497, 2172182, 2174001, 2177972, 2190166, 2206856, 2219486, 2226429,
    2229544, 2231487, 2233338, 2236044, 2256656, 2259212, 2264363, 2268443,
    2268853, 2276258, 2277742, 2279972, 2280649, 2281406, 2281787, 2317335,
    2319095, 2321529, 2325366, 2326432, 2328150, 2342885, 2346627, 2356798,
    2361337, 2363005, 2364673, 2389026, 2391049, 2395406, 2396427, 2397096,
    2398521, 2403003, 2408429, 2410509, 2412080, 2415577, 2417914, 2422106,
    2422699, 2423022, 2437312, 2437616, 2441942, 2442845, 2443114, 2443484,
    2444819, 2445715, 2447366, 2454379, 2457408, 2480495, 2480855, 2481823,
    2483362, 2483708, 2484975, 2486261, 2486410, 2487347, 2488291, 2488702,
    2489166, 2490219, 2492035, 2492660, 2493509, 2493793, 2494079, 2497673,
    2500267, 2504013, 2504458, 2509815, 2510455, 2514041, 2526121, 2536864,
    2606052, 2607072, 2640242, 2641379, 2643566, 2655020, 2666196, 2667093,
    2669723, 2672831, 2676566, 2687172, 2690373, 2692877, 2699494, 2701002,
    2704792, 2708093, 2727426, 2730930, 2747177, 2749479, 2769748, 2776631,
    2777292, 2782093, 2783161, 2786058, 2787622, 2788148, 2790996, 2791124,
    2791270, 2793495, 2794156, 2795169, 2797295, 2799071, 2802426, 2804414,
    2804610, 2807133, 2808304, 2808440, 2814533, 2814860, 2815834, 2817516,
    2823428, 2823750, 2825657, 2834397, 2835271, 2837789, 2840245, 2841315,
    2843684, 2859443, 2860847, 2865351, 2869837, 2870880, 2871525, 2877765,
    2879718, 2883205, 2892201, 2892767, 2894605, 2895154, 2906734, 2909870,
    2910353, 2916936, 2917067, 2927161, 2930766, 2939185, 2948072, 2950826,
    2951358, 2951585, 2963159, 2965783, 2966193, 2966687, 2971356, 2974003,
    2977058, 2978881, 2979186, 2980441, 2981792, 2988304, 2992211, 2992529,
    2999410, 3000134, 3000247, 3000684, 3014705, 3016953, 3017168, 3018349,
    3026506, 3028079, 3032252, 3041632, 3042490, 3045698, 3047690, 3062245,
    3063599, 3063689, 3065424, 3075370, 3085013, 3089624, 3095699, 3100240,
    3109150, 3110669, 3124043, 3124170, 3125729, 3126707, 3127747, 3127925,
    3131574, 3133878, 3134739, 3141823, 3146219, 3160309, 3179701, 3180011,
    3187595, 3188531, 3196217, 3197337, 3201208, 3207743, 3207941, 3208938,
    3216828, 3218198, 3220513, 3223299, 3240683, 3249569, 3250847, 3255030,
    3259280, 3271574, 3272010, 3272562, 3290653, 3291819, 3297495, 3314780,
    3325584, 3337140, 3344393, 3345487, 3347037, 3355925, 3372029, 3376595,
    3379051, 3384352, 3388043, 3388183, 3388549, 3393912, 3394916, 3400231,
    3404251, 3417042, 3424325, 3425413, 3443371, 3444034, 3445777, 3445924,
    3447447, 3447721, 3450230, 3452741, 3457902, 3459775, 3461385, 3467068,
    3476684, 3476991, 3478589, 3481172, 3482405, 3483316, 3485407, 3485794,
    3492542, 3494278, 3495258, 3496892, 3498962, 3527444, 3529860, 3530642,
    3532672, 3534580, 3535780, 3538406, 3544143, 3584254, 3584829, 3590841,
    3594734, 3594945, 3595614, 3598930, 3599486, 3602883, 3617480, 3623198,
    3627232, 3630383, 3633091, 3637318, 3642806, 3649909, 3657121, 3658185,
    3661043, 3662601, 3666591, 3670208, 3673027, 3676483, 3680355, 3690938,
    3691459, 3692522, 3697007, 3706229, 3709823, 3710193, 3710637, 3710721,
    3717622, 3720891, 3721384, 3724870, 3729826, 3733131, 3733281, 3733805,
    3742115, 3743016, 3759954, 3761084, 3763968, 3764736, 3769881, 3770439,
    3770679, 3773504, 3775071, 3775546, 3776460, 3777568, 3777754, 3781244,
    3782006, 3785016, 3786901, 3787032, 3788195, 3788365, 3791053, 3792782,
    3792972, 3793489, 3794056, 3796401, 3803284, 3804744, 3814639, 3814906,
    3825788, 3832673, 3837869, 3838899, 3840681, 3841143, 3843555, 3854065,
    3857828, 3866082, 3868242, 3868863, 3871628, 3873416, 3874293, 3874599,
    3876231, 3877472, 3877845, 3884397, 3887697, 3888257, 3888605, 3891251,
    3891332, 3895866, 3899768, 3902125, 3903868, 3908618, 3908714, 3916031,
    3920288, 3924679, 3929660, 3929855, 3930313, 3930630, 3933933, 3935335,
    3937543, 3938244, 3942813, 3944341, 3947888, 3950228, 3954731, 3956157,
    3958227, 3961711, 3967562, 3970156, 3976467, 3976657, 3977966, 3980874,
    3982430, 3983396, 3991062, 3992509, 3995372, 3998194, 4004767, 4005630,
    4008634, 4009552, 4019541, 4023962, 4026417, 4033901, 4033995, 4037443,
    4039381, 4040759, 4041544, 4044716, 4049303, 4065272, 4067472, 4069434,
    4070727, 4074963, 4081281, 4086273, 4090263, 4099969, 4111531, 4116512,
    4118538, 4118776, 4120489, 4125021, 4127249, 4131690, 4133789, 4136333,
    4141076, 4141327, 4141975, 4146614, 4147183, 4149813, 4152593, 4153751,
    4154565, 4162706, 4179913, 4192698, 4200800, 4201297, 4204238, 4204347,
    4208210, 4209133, 4209239, 4228054, 4229816, 4235860, 4238763, 4239074,
    4243546, 4251144, 4252077, 4252225, 4254120, 4254680, 4254777, 4258138,
    4259630, 4263257, 4264628, 4265275, 4266014, 4270147, 4273569, 4275548,
    4277352, 4285008, 4286575, 4296562, 4310018, 4311004, 4311174, 4317175,
    4325704, 4326547, 4328186, 4330267, 4332243, 4335435, 4336792, 4344873,
    4346328, 4347754, 4350905, 4355338, 4355933, 4356056, 4357314, 4366367,
    4367480, 4370456, 4371430, 4371774, 4372370, 4376876, 4380533, 4389033,
    4392985, 4398044, 4399382, 4404412, 4409515, 4417672, 4418357, 4423845,
    4428191, 4429376, 4435653, 4442312, 4443257, 4447861, 4456115, 4458633,
    4461696, 4462240, 4465501, 4467665, 4476259, 4479046, 4482393, 4483307,
    4485082, 4486054, 4487081, 4487394, 4493381, 4501370, 4505470, 4507155,
    4509417, 4515003, 4517823, 4522168, 4523525, 4525038, 4525305, 4532106,
    4532670, 4536866, 4540053, 4542943, 4548280, 4548362, 4550184, 4552348,
    4553703, 4554684, 4557648, 4560804, 4562935, 4579145, 4579432, 4584207,
    4589890, 4590129, 4591157, 4591713, 4592741, 4596742, 4597913, 4599235,
    4604644, 4606251, 4612504, 4613696, 6359193, 6596364, 6785654, 6794110,
    6874185, 7248320, 7565083, 7579787, 7583066, 7584110, 7590611, 7613480,
    7614500, 7615774, 7684084, 7693725, 7695742, 7697313, 7697537, 7711569,
    7714571, 7714990, 7715103, 7716358, 7716906, 7717410, 7717556, 7718472,
    7718747, 7720875, 7730033, 7734744, 7742313, 7745940, 7747607, 7749582,
    7753113, 7753275, 7753592, 7754684, 7760859, 7768694, 7802026, 7831146,
    7836838, 7860988, 7871810, 7873807, 7875152, 7880968, 7892512, 7920052,
    7930864, 7932039, 9193705, 9229709, 9246464, 9256479, 9288635, 9332890,
    9399592, 9421951, 9428293, 9468604, 9472597, 9835506, 10148035, 10565667,
    11879895, 11939491, 12057211, 12144580, 12267677, 12620546, 12768682,
    12985857, 12998815, 13037406, 13040303, 13044778, 13052670, 13054560,
    13133613, 15075141,
])
