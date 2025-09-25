import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_sprite_frames(image_path):
    """
    소닉 스프라이트 시트에서 각 프레임의 bounding box를 찾는 함수
    """
    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None

    print(f"이미지 크기: {img.shape}")

    # 알파 채널이 있는 경우 (PNG)
    if img.shape[2] == 4:
        # 알파 채널을 사용하여 투명하지 않은 픽셀 찾기
        alpha_channel = img[:, :, 3]
        non_transparent = alpha_channel > 0
    else:
        # RGB 이미지인 경우 배경색(보통 흰색 또는 특정 색상)을 제거
        # 배경색을 자동으로 감지 (가장 많이 사용된 색상)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 각 모서리의 픽셀을 확인하여 배경색 추정
        corners = [img_rgb[0, 0], img_rgb[0, -1], img_rgb[-1, 0], img_rgb[-1, -1]]
        background_color = corners[0]  # 첫 번째 모서리 색상을 배경색으로 가정

        # 배경색과 다른 픽셀들을 찾기
        diff = np.sum(np.abs(img_rgb - background_color), axis=2)
        non_transparent = diff > 30  # 임계값 조정 가능

    # 연결된 컴포넌트 찾기
    non_transparent_uint8 = non_transparent.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_transparent_uint8, connectivity=8)

    # 각 컴포넌트의 bounding box 정보 수집
    sprite_frames = []
    min_area = 100  # 최소 면적 (노이즈 제거)

    for i in range(1, num_labels):  # 0은 배경이므로 제외
        x, y, w, h, area = stats[i]

        if area > min_area:  # 충분히 큰 영역만 선택
            sprite_frames.append({
                'id': i,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'center_x': centroids[i][0],
                'center_y': centroids[i][1]
            })

    # y 좌표로 정렬 후 x 좌표로 정렬 (행별로 정리)
    sprite_frames.sort(key=lambda frame: (frame['y'] // 50, frame['x']))

    return sprite_frames, img, non_transparent

def visualize_bounding_boxes(image_path, sprite_frames, img, non_transparent):
    """
    찾은 bounding box들을 시각화
    """
    plt.figure(figsize=(15, 10))

    # 원본 이미지 표시
    if img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)

    # 각 프레임에 bounding box 그리기
    colors = plt.cm.tab20(np.linspace(0, 1, len(sprite_frames)))

    for i, frame in enumerate(sprite_frames):
        x, y, w, h = frame['x'], frame['y'], frame['width'], frame['height']

        # 사각형 그리기
        rect = plt.Rectangle((x, y), w, h, linewidth=2,
                           edgecolor=colors[i], facecolor='none')
        plt.gca().add_patch(rect)

        # 프레임 번호 표시
        plt.text(x, y-5, f'{i+1}', fontsize=10, color=colors[i],
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                facecolor='white', alpha=0.8))

    plt.title(f'Sonic Sprite Frames - Total: {len(sprite_frames)} frames')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('sonic_sprite_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_actions(sprite_frames):
    """
    스프라이트 프레임들을 행별로 그룹화하여 액션별로 분류
    """
    # y 좌표를 기준으로 행 구분
    rows = {}
    row_tolerance = 30  # 같은 행으로 간주할 y 좌표 차이

    for frame in sprite_frames:
        y = frame['y']
        # 기존 행과 비교
        found_row = False
        for row_y in rows.keys():
            if abs(y - row_y) < row_tolerance:
                rows[row_y].append(frame)
                found_row = True
                break

        if not found_row:
            rows[y] = [frame]

    # 각 행을 y 좌표 순으로 정렬
    sorted_rows = sorted(rows.items())

    print("\n=== 액션별 프레임 분석 ===")
    action_names = [
        "달리기 (기본)",
        "달리기 (고속)",
        "스핀 대시/롤링",
        "점프/공중동작",
        "서있기/대기",
        "브레이킹/멈춤",
        "특수동작 1",
        "특수동작 2",
        "걷기/느린이동",
        "기타 액션"
    ]

    for i, (row_y, frames) in enumerate(sorted_rows):
        action_name = action_names[i] if i < len(action_names) else f"액션 {i+1}"
        frames.sort(key=lambda f: f['x'])  # x 좌표로 정렬

        print(f"\n{i+1}. {action_name} (Y: {row_y})")
        print(f"   프레임 수: {len(frames)}")

        for j, frame in enumerate(frames):
            print(f"   Frame {j+1}: x={frame['x']}, y={frame['y']}, "
                  f"w={frame['width']}, h={frame['height']}")

def main():
    image_path = "sonic-sprite.png"

    print("소닉 스프라이트 시트 분석 시작...")

    # 스프라이트 프레임 찾기
    result = find_sprite_frames(image_path)
    if result is None:
        return

    sprite_frames, img, non_transparent = result

    print(f"\n총 {len(sprite_frames)}개의 스프라이트 프레임을 찾았습니다.")

    # bounding box 정보 출력
    print("\n=== 모든 프레임의 Bounding Box 정보 ===")
    for i, frame in enumerate(sprite_frames):
        print(f"Frame {i+1}: "
              f"x={frame['x']}, y={frame['y']}, "
              f"width={frame['width']}, height={frame['height']}, "
              f"area={frame['area']}")

    # 액션별 분석
    analyze_actions(sprite_frames)

    # 시각화
    visualize_bounding_boxes(image_path, sprite_frames, img, non_transparent)

    return sprite_frames

if __name__ == "__main__":
    frames = main()
