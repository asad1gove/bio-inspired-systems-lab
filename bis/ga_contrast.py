import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img

def calculate_fitness(threshold, img):
    pixels = img.flatten()

    bg_pixels = pixels[pixels <= threshold]
    fg_pixels = pixels[pixels > threshold]

    if len(bg_pixels) == 0 or len(fg_pixels) == 0:
        return 0

    w0 = len(bg_pixels) / len(pixels)
    w1 = len(fg_pixels) / len(pixels)

    mean0 = np.mean(bg_pixels)
    mean1 = np.mean(fg_pixels)

    variance = w0 * w1 * ((mean0 - mean1) ** 2)
    return variance

def genetic_algorithm_segmentation(img, pop_size=20, generations=30, mutation_rate=0.1):
    population = np.random.randint(0, 256, size=pop_size)

    for gen in range(generations):
        fitness_scores = np.array([calculate_fitness(chrom, img) for chrom in population])

        best_idx = np.argmax(fitness_scores)
        best_threshold = population[best_idx]
        best_fit = fitness_scores[best_idx]

        print(f"Generation {gen+1}/{generations} | Best Threshold: {best_threshold} | Fitness: {best_fit:.2f}")

        if np.sum(fitness_scores) == 0:
            probs = np.ones(pop_size) / pop_size
        else:
            probs = fitness_scores / np.sum(fitness_scores)

        selected_indices = np.random.choice(np.arange(pop_size), size=pop_size, p=probs)
        population = population[selected_indices]

        next_generation = []
        for i in range(0, pop_size, 2):
            p1, p2 = population[i], population[(i+1) % pop_size]

            p1_bin = format(p1, '08b')
            p2_bin = format(p2, '08b')

            cp = np.random.randint(1, 7)
            c1_bin = p1_bin[:cp] + p2_bin[cp:]
            c2_bin = p2_bin[:cp] + p1_bin[cp:]

            next_generation.extend([int(c1_bin, 2), int(c2_bin, 2)])

        population = np.array(next_generation)

        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                bin_str = list(format(population[i], '08b'))
                bit_to_flip = np.random.randint(0, 8)
                bin_str[bit_to_flip] = '1' if bin_str[bit_to_flip] == '0' else '0'
                population[i] = int("".join(bin_str), 2)

    final_fitness = np.array([calculate_fitness(chrom, img) for chrom in population])
    optimal_threshold = population[np.argmax(final_fitness)]
    return optimal_threshold

if __name__ == "__main__":
    IMAGE_PATH = "thisbepic.jpg"

    try:
        gray_img = load_image(IMAGE_PATH)
        print("Image successfully loaded.")

        print("Starting Genetic Algorithm optimization...")
        optimal_thresh = genetic_algorithm_segmentation(gray_img, pop_size=30, generations=20)
        print(f"\nOptimization complete! Optimal Threshold Found: {optimal_thresh}")

        _, segmented_img = cv2.threshold(gray_img, optimal_thresh, 255, cv2.THRESH_BINARY)

        cv2.imwrite("output_segmented.png", segmented_img)
        print("Segmented image saved as 'output_segmented.png'")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(gray_img, cmap='gray')
        plt.title('Initial Grayscale Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_img, cmap='gray')
        plt.title(f'Segmented Image (Threshold: {optimal_thresh})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")