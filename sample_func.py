import numpy as np


def get_sample(gts, POINT_NUM_GT,  types=0, step=0, sample_weight=None, grid_points=None, grid_sizes=None):
    index = np.random.choice(gts.shape[0], POINT_NUM_GT, replace=True)
    if types == 0:
        noise = gts[index[:3 * POINT_NUM_GT / 8]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[3*POINT_NUM_GT//8, 3])
        noise2 = gts[index[3 * POINT_NUM_GT / 8:7 * POINT_NUM_GT / 8]] + 0.5* 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT//2, 3])
        noise3 = gts[index[7 * POINT_NUM_GT / 8:15 * POINT_NUM_GT / 16]]
        noise4 = noise3 + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1*POINT_NUM_GT//16, 3]) # np.random.uniform(-2.0/VOX_SIZE, 2.0/VOX_SIZE, size=[POINT_NUM_GT/16, 3])
        noise = np.concatenate([noise, noise2, noise3, noise4], 0)
    elif types == 1:
        noise = gts[index] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
    elif types == 2:
        noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
    elif types == 3:
        noise = gts[index]
    elif types == 5:
        noise1 = gts[index[:3 * POINT_NUM_GT / 4]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[3 * POINT_NUM_GT / 4, 3])
        noise2 = gts[index[:1 * POINT_NUM_GT / 4]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 4, 3])
        noise = np.concatenate([noise1, noise2])
    elif types == 6:
        noise1 = gts[index[:3 * POINT_NUM_GT / 4]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[3 * POINT_NUM_GT / 4, 3])
        noise2 = gts[index[:1 * POINT_NUM_GT / 4]] # + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 4, 3])
        noise = np.concatenate([noise1, noise2])
    elif types == 7:
        noise1 = gts[index[:3 * POINT_NUM_GT / 4]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[3 * POINT_NUM_GT / 4, 3])
        noise2 = gts[index[:1 * POINT_NUM_GT / 4]] + 0.3 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 4, 3])
        noise = np.concatenate([noise1, noise2]) 
    elif types == 8:
        noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
        noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
        noise = np.concatenate([noise1, noise2])
    elif types == 9:
        if step < 500:
            noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT // 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT // 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT // 2:]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT // 2, 3])
            noise = np.concatenate([noise1, noise2])
    # for regular grad calculate
    elif types == 10:
        noise1 = gts[index[:3 * POINT_NUM_GT // 8]] + 0.01 * 0.027 * np.random.normal(0.0, 1.0, size=[3 * POINT_NUM_GT // 8, 3])
        noise2 = gts[index[3 * POINT_NUM_GT // 8:]] + 0.02 * 0.027 * np.random.normal(0.0, 1.0, size=[5 * POINT_NUM_GT // 8, 3])
        noise = np.concatenate([noise1, noise2])
    elif types == 11:
        if step < 6000:
            noise = gts[index] + 1.8 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 1.8 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise = np.concatenate([noise1, noise2])
    elif types == 12:
        if step < 6000:
            noise = gts[index] + 2.8 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 2.8 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise = np.concatenate([noise1, noise2])
    elif types == 13:
        noise = np.random.uniform(-0.55, 0.55, size=[POINT_NUM_GT, 3])
    # for 128 
    elif types == 14:
        if step < 6000:
            noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.05 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise = np.concatenate([noise1, noise2])
    elif types == 15:
        if step < 6000:
            noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.01 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise = np.concatenate([noise1, noise2])
    elif types == 16:
        if step < 6000:
            noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise1 = gts[index[:1 * POINT_NUM_GT / 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise2 = gts[index[1 * POINT_NUM_GT / 2:]] + 0.15 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * POINT_NUM_GT / 2, 3])
            noise = np.concatenate([noise1, noise2])
    elif types == 17:
        if step < 3000:
            noise = gts[index] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        elif step < 6000:
            noise = gts[index] + 0.3 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        elif step < 9000:
            noise = gts[index] + 0.2 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
        else:
            noise = gts[index] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[POINT_NUM_GT, 3])
    elif types == 18:
        if sample_weight is None:
            n = POINT_NUM_GT
        else:
            n = POINT_NUM_GT - sample_weight.shape[0]
            if n < 0:
                return sample_weight[:POINT_NUM_GT]
        if step < 6000:
            noise = gts[index[:n]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[n, 3])
        else:
            noise1 = gts[index[:1 * n / 2]] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * n / 2, 3])
            noise2 = gts[index[:1 * n / 2]] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[1 * n / 2, 3])
            noise = np.concatenate([noise1, noise2])
        if sample_weight is not None:
            noise = np.concatenate([noise, sample_weight], 0)
        if noise.shape[0] < POINT_NUM_GT:
            k = POINT_NUM_GT - noise.shape[0]
            tnoise = gts[:k] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[k, 3])
            noise = np.concatenate([noise, tnoise], 0)
    elif types == 19:
        if sample_weight is None:
            n = POINT_NUM_GT
        else:
            n = POINT_NUM_GT - sample_weight.shape[0]
            if n < 0:
                return sample_weight[:POINT_NUM_GT]
        if step < 3000:
            noise = gts[:n] + 0.5 * 0.027 * np.random.normal(0.0, 1.0, size=[n, 3])
        elif step < 6000:
            noise = gts[:n] + 0.3 * 0.027 * np.random.normal(0.0, 1.0, size=[n, 3])
        elif step < 9000:
            noise = gts[:n] + 0.2 * 0.027 * np.random.normal(0.0, 1.0, size=[n, 3])
        else:
            noise = gts[:n] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[n, 3])
        if sample_weight is not None:
            noise = np.concatenate([noise, sample_weight], 0)        
        if noise.shape[0] < POINT_NUM_GT:
            k = POINT_NUM_GT - noise.shape[0]
            tnoise = gts[:k] + 0.1 * 0.027 * np.random.normal(0.0, 1.0, size=[k, 3])
            noise = np.concatenate([noise, tnoise], 0)
    elif types == 20:
        index = np.random.choice(grid_points.shape[0], POINT_NUM_GT, replace=True)
        grid_gts = grid_points[index]
        grid_sizes = grid_sizes[index][:, None]
        noise = grid_gts + grid_sizes * np.random.uniform(-1.0, 1.0, size=[POINT_NUM_GT, 3])
    return noise


def bigger(p, size):
    bd = 1
    p = p * ((size-1) / 2.0 / bd) + (size - 1) / 2.0
    return p


def init_sphere(radius, size):
    grid = np.ones([size, size, size])
    indexes = np.where(grid > 0)
    points = np.concatenate([indexes[0][:, None], indexes[1][:, None], indexes[2][:, None]], 1)
    center = np.array([[size/2, size/2, size/2]])
    dis = np.sqrt(np.sum((points - center)**2, 1)) - radius * size
    grid = np.reshape(dis, [size, size, size])
    # grid /= size
    return grid


def unique_3d(grid, weight=None):
    grid_single = grid[:, 0] * np.sqrt(2) + grid[:, 1] * np.sqrt(3) + grid[:, 2] * np.sqrt(5)
    _, indices = np.unique(grid_single, return_index=True)
    if weight is None:
        return grid[indices]
    return grid[indices], weight[indices]


def get_index(points, level):
    all_points = []
    all_weights = []
    index = 0
    for i in range(-level, level):
        for j in range(-level, level):
            for k in range(-level, level):
                temp = points.copy()
                temp[:, 0] += i
                temp[:, 1] += j
                temp[:, 2] += k
                if i <= 15 or j <= 15 or k <= 15:
                    ii = 1
                    jj = 0
                    kk = 0
                    w = 1
                else:
                    ii = i
                    jj = j
                    kk = k
                    w = 200 / 21.0
                ii = i
                jj = j
                kk = k
                weight = w * (abs(ii)+abs(jj)+abs(kk)) + 1 # / (3.0 * level)
                all_points.append(temp)
                all_weights.append(weight * np.ones([temp.shape[0]]))
                index += 1
                if index % 20 == 0:
                    a = np.concatenate(all_points, 0)
                    b = np.concatenate(all_weights, 0)
                    a, b = unique_3d(a, b)
                    all_points = [a]
                    all_weights = [b]
    all_points = np.concatenate(all_points, 0)
    all_weights = np.concatenate(all_weights, 0)
    all_points, all_weights = unique_3d(all_points, all_weights)
    return all_points, all_weights


def init_smooth_grid_index(pc, size, level):
    pc = np.int64(unique_3d(np.round(pc)))
    train_index, train_smooth_weight = get_index(pc, level) 
    test_index, _  = get_index(pc, 3)
    return train_index, test_index, train_smooth_weight
