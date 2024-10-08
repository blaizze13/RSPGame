import random
import numpy as np
import pymunk
import pymunk.pyglet_util
import math

space = pymunk.Space()

def createBody(x, y, shape, *shapeArgs):
    body = pymunk.Body()
    body.position = x, y
    s = shape(body, *shapeArgs)
    s.mass = 1
    s.friction = 1
    space.add(body, s)
    return s

s0 = createBody(300, 300, pymunk.Poly, ((-20, -5), (-20, 5), (20, 15), (20, -15)))
s0.score = 0
s3 = createBody(200, 300, pymunk.Poly, ((-20, -5), (-20, 5), (20, 15), (20, -15)))
s3.color = (0, 255, 0, 255)
s3.score = 0

s3.body.Q = np.zeros((3, 2))
s3.body.action = 0

def update_Q_table(state, action, reward, next_state):
    max_future_q = np.max(s3.body.Q[next_state])
    s3.body.Q[state, action] += alpha * (reward + gamma * max_future_q - s3.body.Q[state, action])

alpha = 0.1
gamma = 0.9

def strategy2(b=s3.body):
    v = 100
    a = b.angle
    b.velocity = v * math.cos(a), v * math.sin(a)
    x, y = b.position

    R1 = getDist(x, y, s1.body.position[0], s1.body.position[1])
    R2 = getDist(x, y, S2[0].body.position[0], S2[0].body.position[1])

    if canvas.frame % 10 == 0:
        inS = inSector(s1.body.position[0], s1.body.position[1], x, y, 100, a)
        inS2 = inSector(S2[0].body.position[0], S2[0].body.position[1], x, y, 100, a)

        if inS:
            state = 1
            reward = 1 if b.action == 0 else -1
        elif inS2:
            state = 2
            reward = -1 if b.action == 0 else 1
        else:
            state = 0
            reward = 0

        update_Q_table(state, b.action, reward, state)
        
        if random.random() < 0.1:
            b.action = random.choice([0, 1])
        else:
            b.action = np.argmax(b.Q[state])

        if b.action:
            b.angle = 2 * math.pi * random.random()

        if R > 180:
            b.angle = getAngle(x, y, 350, 250)

def scr(s, s0, s3, p=1):
    bx, by = s.body.position
    s0x, s0y = s0.body.position
    s3x, s3y = s3.body.position
    if not inCircle(bx, by, 350, 250, 180):
        if getDist(bx, by, s0x, s0y) < getDist(bx, by, s3x, s3y):
            s0.score = s0.score + p
        else:
            s3.score = s3.score + p
        s.body.position = random.randint(200, 400), random.randint(200, 300)

def score():
    scr(s1, s0, s3)
    for s in S2:
        scr(s, s0, s3, p=-1)

def draw(canvas):
    canvas.clear()
    fill(0, 0, 0, 1)
    text("%i %i" % (s0.score, s3.score), 20, 20)
    nofill()
    ellipse(350, 250, 350, 350, stroke=Color(0))
    strategy2()
    score()
    simFriction()
    space.step(0.02)
    space.debug_draw(draw_options)

canvas.size = 700, 500
canvas.run(draw)
