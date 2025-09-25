from pico2d import *

open_canvas(800, 600)
image = load_image('sonic-sprite.png')

sprite = (
    ( (0, 0, 100, 100), (0, 0, 100, 100))
)

def play_animation(action):
    for frame in action:
        clear_canvas()
        image.clip_draw(frame[0], frame[1], frame[2], frame[3], 400, 300, frame[2]*3, frame[3]*3)
        update_canvas()
        delay(0.1)
    pass

while True:
    # 모든 애니메이션 차례로 재생
    for action in sprite:
        for i in range(5):
            play_animation(action)
        delay(1)
    pass

close_canvas()