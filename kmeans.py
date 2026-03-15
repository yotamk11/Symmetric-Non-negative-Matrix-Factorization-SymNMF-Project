import sys

def euclidean_distance(point1, point2):
    s = 0.0
    for i in range(len(point1)):
        s += (point1[i] - point2[i]) ** 2
    return s ** 0.5

def read_data_from_stdin():
    data = []
    dim = None

    for line in sys.stdin:
        line = line.strip()

        if line == '':
            continue

        if line.endswith(','):
            print("An Error Has Occurred")
            sys.exit(1)

        parts = line.split(',')

        for p in parts:
            if p == '':
                print("An Error Has Occurred")
                sys.exit(1)

        try:
            point = [float(p) for p in parts]
        except ValueError:
            print("An Error Has Occurred")
            sys.exit(1)

        if dim is None:
            dim = len(point)
            if dim == 0:
                print("An Error Has Occurred")
                sys.exit(1)
        else:
            if len(point) != dim:
                print("An Error Has Occurred")
                sys.exit(1)

        data.append(point)
    return data


# assumes params are valid
def kmeans_alg(K, data, iter=400, epslion=0.001):
    centroids = data[:K]
    close_enough = False

    while iter > 0 and close_enough is False:
        clusters = [[] for _ in range(K)]
        for point in data:
            min_dist = float('inf')
            min_index = -1
            for i in range(K):
                dist = euclidean_distance(point, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            clusters[min_index].append(point)

        new_centroids = []
        for i in range(K):
            if len(clusters[i]) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroid = [sum(dim) / len(clusters[i]) for dim in zip(*clusters[i])]
                new_centroids.append(new_centroid)

        close_enough = True
        for i in range(K):
            if euclidean_distance(centroids[i], new_centroids[i]) >= epslion:
                close_enough = False
                break

        centroids = new_centroids
        iter -= 1

    return centroids

def print_centroids(centroids):
    for centroid in centroids:
        print(','.join(f"{coord:.4f}" for coord in centroid))

def main():
    
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        K = int(sys.argv[1])
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

    if len(sys.argv) == 3:
        try:
            iter = int(sys.argv[2])
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)
    else:
        iter = 400

    # Validate iter range: 1 < iter < 800
    if not (1 < iter < 800):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    data = read_data_from_stdin()

    # Validate input is not empty
    if len(data) == 0:
        print("An Error Has Occurred")
        sys.exit(1)
    N = len(data)

    # Validate K range: 1 < K < N
    if not (1 < K < N):
        print("Incorrect number of clusters!")
        sys.exit(1)

    centroids = kmeans_alg(K, data, iter)
    print_centroids(centroids)

if __name__ == "__main__":
    main()