import random
from assignment7.helpers import euclidean_distance, mean, vectors_equal, standard_deviation


def k_means(dataset, k):
    # Initialize random centroids
    indexes = list(range(len(dataset)))
    random.shuffle(indexes)
    indexes = indexes[:k]
    centroids = [dataset[i][:-1] for i in indexes]

    while True:
        # Find closest centroids of each point in dataset
        closest_centroids = []
        for point in dataset:
            closest = 0
            min_distance = float('inf')
            for c in range(len(centroids)):
                dist = euclidean_distance(point[:-1], centroids[c])
                if dist < min_distance:
                    closest = c
                    min_distance = dist
            closest_centroids.append(closest)

        # Calculate new centroids based on the mean
        new_centroids = list()
        for c in range(len(centroids)):
            close_points = [dataset[i][:-1] for i in range(len(closest_centroids)) if closest_centroids[i] == c]
            if len(close_points) == 0:
                new_centroids.append(centroids[c])
            else:
                mean_point = list()
                for j in range(len(close_points[0])):
                    cpj = [row[j] for row in close_points]
                    mean_point.append(mean(cpj))
                new_centroids.append(mean_point)

        # Check if points changed
        centroids_changed = False
        for c in range(len(new_centroids)):
            if not vectors_equal(centroids[c], new_centroids[c]):
                centroids_changed = True
                break

        # End if centroids didn't change
        if not centroids_changed:
            break

        # Update centroids
        centroids = new_centroids

    return new_centroids
