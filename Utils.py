import networkx as nx
import matplotlib.pyplot as plt
import random
import osm_graph as og
import numpy as np
import math
import torch
import decimal


def random_walk_sample(G, sample_num, path_length, start):
    rw = []
    path = [[]] * sample_num
    used_node = [[]] * sample_num
    for i in range(sample_num):
        path[i] = [start]
        used_node[i] = [start]
        path_pos = []
        test_repeat = True
        ite_count = 0
        while (test_repeat):
            while (len(path[i]) <= path_length):
                cur_node = path[i][-1]
                possibleNode = list(set(list(G.adj[cur_node])).difference(set(used_node[i])))
                if (len(possibleNode) > 0):
                    rd = random.sample(possibleNode, 1)
                    used_node[i].extend(rd)
                    # used_node.extend(rd)
                    # path_pos.append(get_posdiff(G, path[i][-1], path[i][-2]))
                    ld = latlon_distance(get_pos(G, rd[0]), get_pos(G, path[i][-1]))
                    if (abs(ld[0]) <= 100 and abs(ld[1]) <= 100):
                        path[i].extend(rd)
                        ld_index = get_vocabindex(ld)
                        path_pos.append(ld_index)
                else:
                    # print("采样数不足！重新采样")
                    return [0]
                    # return random_walk_sample(G, sample_num, path_length, start)
            test_repeat = False
            for j in range(i):
                if (path[i] == path[j]):
                    if (ite_count < 3):
                        ite_count += 1
                        test_repeat = True
                    else:
                        return [1]
                    break
        path_pos.reverse()
        rw.append(path_pos)
    return (rw)


def get_pos(G, node):
    att = G.nodes[node]
    pos = [att['lat'], att['lon']]
    return pos


def get_posdiff(G, node1, node2):
    pos1 = get_pos(G, node1)
    pos2 = get_pos(G, node2)
    posdiff = [pos2[i] - pos1[i] for i in range(2)]
    return posdiff


def sort_counterclockwise(points):
    angles = [math.atan2(y, x) for x, y in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points


def latlon_distance(homePos, destinationPos):
    R = 6371e3
    homeLatitude = homePos[0]
    homeLongitude = homePos[1]
    destinationLatitude = destinationPos[0]
    destinationLongitude = destinationPos[1]

    rlat1 = homeLatitude * (math.pi / 180)
    rlat2 = destinationLatitude * (math.pi / 180)
    rlon1 = homeLongitude * (math.pi / 180)
    rlon2 = destinationLongitude * (math.pi / 180)
    dlat = (destinationLatitude - homeLatitude) * (math.pi / 180)
    dlon = (destinationLongitude - homeLongitude) * (math.pi / 180)

    # Haversine formula to find distance
    a = (math.sin(dlat / 2) * math.sin(dlat / 2)) + (
            math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon / 2) * math.sin(dlon / 2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in metres
    distance = R * c

    # Formula for bearing
    y = math.sin(rlon2 - rlon1) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(rlon2 - rlon1)

    # Bearing in radians
    bearing = math.atan2(y, x)
    # bearingDegrees = bearing * (180 / math.pi)
    # out = [distance, bearingDegrees]

    distance_x = distance * math.sin(bearing)
    distance_y = distance * math.cos(bearing)

    # decimal.getcontext().rounding = "ROUND_HALF_UP"
    # distance_x = int(decimal.Decimal(str(distance_x)).quantize(decimal.Decimal("0")))
    # distance_y = int(decimal.Decimal(str(distance_y)).quantize(decimal.Decimal("0")))
    distance_x = round(distance_x)
    distance_y = round(distance_y)

    out = [distance_x, distance_y]

    return out


def get_vocabindex(dis, length=100):
    index = (length * 2 + 1) * (dis[0] + length) + dis[1] + length
    return index


def get_osm(index):
    area = 10
    dist = 500 * math.sqrt(area)
    osm1 = (37.5698, 126.9833)  # 首尔
    osm2 = (35.8203, 139.9241)  # 东京
    osm3 = (34.7502, 135.6030)  # 大阪
    osm4 = (40.7193, -73.8789)  # 纽约
    osm5 = (34.0478, -118.2482)  # 洛杉矶
    osm6 = (33.7989, -117.9065)  # 安纳海姆
    osm7 = (48.8687, 2.3845)  # 巴黎
    osm8 = (51.5260, -0.0689)  # 伦敦
    osm9 = (52.5181, 13.3879)  # 柏林
    osm10 = (22.3212, 114.1769)  # 香港
    osm11 = (31.2304, 121.4750)  # 上海
    osmlist = [osm1, osm2, osm3, osm4, osm5, osm6, osm7, osm8, osm9, osm10, osm11]
    ot = osmlist[index - 1]
    osm = og.OsmGraph(ot[0], ot[1], dist=dist)
    return osm


def get_lcs(G):
    largest = max(nx.connected_components(G), key=len)
    largest_connected_subgraph = G.subgraph(largest)
    return largest_connected_subgraph


def get_vocab(length):
    vocab = {}
    for i in range(-length, length + 1):
        for j in range(-length, length + 1):
            key = str([i, j])
            value = (length * 2 + 1) * (i + length) + j + length
            vocab.update({key: value})
    vocab.update({'end': len(vocab)})
    return vocab


def get_dataset(G, K, L):
    encset = []
    decset = []
    validlen = []
    maxrec = 5
    sort_md = sorted(G.degree, key=lambda x: x[1], reverse=True)
    maxdegree = sort_md[0][1]
    for node in G.nodes:
        nei = list(G.adj[node])
        if (len(nei) > 0):
            enclist = random_walk_sample(G, K, L, node)
            count = 0
            while (enclist == [0]):
                if (count < maxrec):
                    enclist = random_walk_sample(G, K, L, node)
                    count += 1
                else:
                    break
            if (len(enclist) != 1):
                encset.append(enclist)
                declist0 = []
                for i in range(len(nei)):
                    # declist0.append(get_posdiff(G, node, nei[i]))
                    ld = latlon_distance(get_pos(G, node), get_pos(G, nei[i]))
                    if (abs(ld[0]) <= 100 and abs(ld[1]) <= 100):
                        declist0.append(ld)
                ccw = sort_counterclockwise(declist0)
                # declist = [[0, 0, 0]] * (maxdegree + 1)
                # for i in range(len(ccw)):
                #     if (i != len(ccw) - 1):
                #         ccw[i].append(1)
                #     else:
                #         ccw[i].append(0)
                #     declist[i] = ccw[i]
                declist = [0] * (maxdegree + 1)
                for i in range(len(ccw)):
                    declist[i] = get_vocabindex(ccw[i])
                declist[len(ccw)] = 40401
                decset.append(declist)
                validlen.append(len(ccw))
    encset = torch.as_tensor(encset, dtype=torch.long)
    decset = torch.as_tensor(decset, dtype=torch.long)
    validlen = torch.as_tensor(validlen, dtype=torch.int32)

    return encset, decset, validlen


