import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from option import opt

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None
    return image / 255.0

def read_image_rgb(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None
    return image

def extract_r_channel(image):
    return image[:, :, 2] / 255.0  

def compute_gradient(image):
    gradient_x = np.gradient(image, axis=1)
    gradient_y = np.gradient(image, axis=0)
    return gradient_x, gradient_y

def l1_norm_gradient(gradient_x, gradient_y):
    return np.abs(gradient_x) + np.abs(gradient_y)

def apply_gaussian_filter(image, sigma):
    return gaussian_filter(image, sigma=sigma)


def huber_loss_np(x, delta=0.01):
    abs_x = np.abs(x)
    quadratic = 0.5 * x ** 2
    linear = delta * (abs_x - 0.5 * delta)
    return np.where(abs_x <= delta, quadratic, linear)

def huber_gradient_np(x, delta=0.01):
    return np.where(np.abs(x) <= delta, x, delta * np.sign(x))

def optimize1(I_r, S_I, sigma, rho, max_iter=300, tol=1e-4, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = np.zeros_like(S_I), np.zeros_like(S_I)
    t = 0
    prev_loss = float('inf')
    delta = 0.01

    for i in range(max_iter):
        t += 1
        grad_x, grad_y = compute_gradient(S_I)
        grad_S_I = l1_norm_gradient(grad_x, grad_y)

        I_r_blurred = apply_gaussian_filter(I_r, sigma)
        weighted_grad = I_r_blurred * grad_S_I

        first_term_grad = -2 * (I_r - S_I) / rho

        # 使用Huber梯度
        grad_huber_x = huber_gradient_np(grad_x, delta)
        grad_huber_y = huber_gradient_np(grad_y, delta)
        grad_huber_total = I_r_blurred * (grad_huber_x + grad_huber_y)

        grad_total = first_term_grad + grad_huber_total

        m = beta1 * m + (1 - beta1) * grad_total
        v = beta2 * v + (1 - beta2) * (grad_total**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        S_I -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # 使用Huber损失
        loss_reg = np.sum(huber_loss_np(I_r_blurred * grad_S_I, delta))
        loss = (1 / rho) * np.linalg.norm(I_r - S_I, 'fro')**2 + loss_reg
        print(f"I Iter {i+1}/{max_iter}, Loss: {loss}")

        if np.abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {i+1} with Loss: {loss}, ΔLoss: {abs(prev_loss - loss)}")
            break
        prev_loss = loss

    return S_I

def optimize2(G_r, S_G, sigma, rho, max_iter=300, tol=1e-4, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = np.zeros_like(S_G), np.zeros_like(S_G)
    t = 0
    prev_loss = float('inf')
    delta = 0.01

    for i in range(max_iter):
        t += 1
        grad_x, grad_y = compute_gradient(S_G)
        grad_S_G = l1_norm_gradient(grad_x, grad_y)

        G_r_blurred = apply_gaussian_filter(G_r, sigma)
        weighted_grad = G_r_blurred * grad_S_G

        first_term_grad = -2 * (G_r - S_G) / rho

        grad_huber_x = huber_gradient_np(grad_x, delta)
        grad_huber_y = huber_gradient_np(grad_y, delta)
        grad_huber_total = G_r_blurred * (grad_huber_x + grad_huber_y)

        grad_total = first_term_grad + grad_huber_total

        m = beta1 * m + (1 - beta1) * grad_total
        v = beta2 * v + (1 - beta2) * (grad_total**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        S_G -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        loss_reg = np.sum(huber_loss_np(G_r_blurred * grad_S_G, delta))
        loss = (1 / rho) * np.linalg.norm(G_r - S_G, 'fro')**2 + loss_reg
        print(f"G Iter {i+1}/{max_iter}, Loss: {loss}")

        if np.abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {i+1} with Loss: {loss}, ΔLoss: {abs(prev_loss - loss)}")
            break
        prev_loss = loss

    return S_G

def de_enhance_mapping(S_I_star, S_G_star, window_size=3):
    E = np.zeros_like(S_I_star)
    half_window = window_size // 2
    
    for i in range(half_window, S_I_star.shape[0] - half_window):
        for j in range(half_window, S_I_star.shape[1] - half_window):
            region_S_I_star = S_I_star[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            region_S_G_star = S_G_star[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            max_S_I_star = np.max(region_S_I_star)
            max_S_G_star = np.max(region_S_G_star)
            E[i, j] = max_S_G_star / (max_S_I_star + 1e-6)
    
    return E

def estimate_optical_flow_tvl1(I_r_star, G_r):
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(I_r_star, G_r, None)
    return flow

def flow_to_color(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_image(image, flow):
    h, w = image.shape[:2]
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.array(flow_map).transpose(1, 2, 0) + flow
    remap = cv2.remap(image, flow_map[..., 0].astype(np.float32),
                      flow_map[..., 1].astype(np.float32),
                      interpolation=cv2.INTER_LINEAR)
    return remap

def align_images(I_r, S_I, G_r, S_G, G, rho, sigma, save_path,hazy_path, De_enhanced_path):
    # 计算并保存 S_I_star
    S_I_star = optimize1(I_r, S_I, sigma, rho)
    # cv2.imwrite(os.path.join(save_path, "S_I_star.png"), (S_I_star * 255).astype(np.uint8))
    
    # 计算并保存 S_G_star
    S_G_star = optimize2(G_r, S_G, sigma, rho)
    # cv2.imwrite(os.path.join(save_path, "S_G_star.png"), (S_G_star * 255).astype(np.uint8))
    
    # 计算 E map
    E = de_enhance_mapping(S_I_star, S_G_star, window_size=3)
    
    # 处理 E map 中的无穷大和 NaN 值
    E = np.nan_to_num(E, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 保存原始的 E map（作为灰度图）
    E_gray = np.clip(E, 0, 1)  # 限制值在 [0,1] 范围内
    # cv2.imwrite(os.path.join(save_path, "E_map_gray.png"), (E_gray * 255).astype(np.uint8))
    
    # 计算去增强后的图像 I_r_star
    I_r_star = E * I_r
    
    # 处理 I_r_star 中的无穷大和 NaN 值
    I_r_star = np.nan_to_num(I_r_star, nan=0.0, posinf=1.0, neginf=0.0)
    I_r_star = np.clip(I_r_star, 0, 1)  # 限制值在 [0,1] 范围内
    basename = os.path.splitext(os.path.basename(hazy_path))[0]

    # 保存去增强后的图像（作为灰度图）
    cv2.imwrite(os.path.join(De_enhanced_path, f"{basename}.png"), (I_r_star * 255).astype(np.uint8))

    # 为了更好的可视化，也保存一个彩色的热力图版本
    E_heatmap = cv2.applyColorMap((E_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join(save_path, "E_map_heatmap.png"), E_heatmap)

    # 转换为 float32 用于光流计算
    I_r_star = I_r_star.astype(np.float32)
    G_r = G_r.astype(np.float32)

    # 计算并保存光流
    flow = estimate_optical_flow_tvl1(I_r_star, G_r)

    flow_vis = flow_to_color(flow)
    # cv2.imwrite(os.path.join(save_path, "flow_visualization.png"), flow_vis)

    # 生成并返回对齐图像
    aligned_image = warp_image(G, flow)
    # cv2.imwrite(os.path.join(save_path, "aligned_G.png"), aligned_image)
    # aligned_G_r = warp_image(G_r, flow_1)
    aligned_G_r = warp_image(G_r, flow)

    cv2.imwrite(os.path.join(save_path, f"{basename}.png"), (aligned_G_r*255).astype(np.uint8))
    # aligned_G_r = warp_image(G_r, flow)
    # cv2.imwrite(os.path.join(save_path, "flow_Gr.png"), aligned_G_r)
    return aligned_image

def process_image_pair(hazy_path, clear_path, hazy_r_s_path, clear_r_s_path, result_path, rho, sigma):
    # 创建保存中间结果的文件夹
    result_dir = os.path.dirname(result_path)
    basename = os.path.splitext(os.path.basename(hazy_path))[0]
    intermediate_dir = os.path.join(result_dir, f"C_r_star")
    os.makedirs(intermediate_dir, exist_ok=True)

    De_enhanced_dir = os.path.join(result_dir, f"S_r_q")
    os.makedirs(De_enhanced_dir, exist_ok=True)

    # 读取输入图像
    I_r = extract_r_channel(read_image_rgb(hazy_path))
    G_r = extract_r_channel(read_image_rgb(clear_path))
    G = read_image_rgb(clear_path)
    S_I = read_image(hazy_r_s_path)
    S_G = read_image(clear_r_s_path)

    if I_r is None or G_r is None or S_I is None or S_G is None or G is None:
        print(f"Skipping {hazy_path} and {clear_path} due to read error.")
        return

    if I_r.shape != G_r.shape or S_I.shape != S_G.shape:
        print(f"Skipping {hazy_path} and {clear_path} due to dimension mismatch.")
        return

    # # 保存输入的 R 通道图像（作为灰度图）
    # cv2.imwrite(os.path.join(intermediate_dir, "I_r.png"), (I_r * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(intermediate_dir, "G_r.png"), (G_r * 255).astype(np.uint8))
    
    # # 保存结构图像
    # cv2.imwrite(os.path.join(intermediate_dir, "S_I.png"), (S_I * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(intermediate_dir, "S_G.png"), (S_G * 255).astype(np.uint8))

    # 进行图像对齐并保存结果
    aligned_image = align_images(I_r, S_I, G_r, S_G, G, rho, sigma, intermediate_dir,hazy_path, De_enhanced_dir)
    cv2.imwrite(result_path, aligned_image)
    print(f"Saved result and intermediates in: {intermediate_dir}")

def process_folder(base_folder, rho, sigma):
    hazy_folder = os.path.join(base_folder, "hazy")
    clear_folder = os.path.join(base_folder, "clear")
    hazy_r_s_folder = os.path.join(base_folder, "hazy_r_s")
    clear_r_s_folder = os.path.join(base_folder, "clear_r_s")
    result_folder = os.path.join(base_folder, "results")

    # clear_save_folder = os.path.join(result_folder, "aligned_Gr")
    # dehanced_save_folder = os.path.join(result_folder, "DeEnhanced")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # if not os.path.exists(clear_save_folder):
    #     os.makedirs(clear_save_folder)
    # if not os.path.exists(dehanced_save_folder):
    #     os.makedirs(dehanced_save_folder)

    hazy_files = sorted(os.listdir(hazy_folder))
    clear_files = sorted(os.listdir(clear_folder))

    for hazy_file, clear_file in zip(hazy_files, clear_files):
        hazy_path = os.path.join(hazy_folder, hazy_file).replace("\\", "/")
        clear_path = os.path.join(clear_folder, clear_file).replace("\\", "/")
        hazy_r_s_path = os.path.join(hazy_r_s_folder, hazy_file).replace("\\", "/")
        clear_r_s_path = os.path.join(clear_r_s_folder, clear_file).replace("\\", "/")
        result_path = os.path.join(result_folder, hazy_file).replace("\\", "/")

        if os.path.isfile(hazy_path) and os.path.isfile(clear_path):
            print(f"Processing pair: {hazy_file} and {clear_file}")
            process_image_pair(hazy_path, clear_path, hazy_r_s_path, clear_r_s_path, result_path, rho, sigma)
        else:
            print(f"Skipping: {hazy_file} or {clear_file} not found.")

if __name__ == "__main__":
    base_folder = opt.base_dir  # 替换为您的文件夹路径
    rho = 0.01  # 正则化参数
    sigma = 10  # 高斯核标准差
    process_folder(base_folder, rho, sigma)