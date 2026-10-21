import numpy as np

from smart_grid.config import load_config
from smart_grid.envs.delay import DelayBuffer


def make_buffer():
    buffer = DelayBuffer(load_config()["delay"], np.random.default_rng(0))
    buffer.reset(np.array([0.0]))
    return buffer


def test_delivery_age_and_no_future_leak():
    buffer = make_buffer()
    buffer.submit(1, np.array([99.0]), delay_steps=3)
    measurement, age = buffer.receive(3)
    assert measurement[0] == 0 and age == 3
    measurement, age = buffer.receive(4)
    assert measurement[0] == 99 and age == 3


def test_out_of_order_packet_cannot_overwrite_newer_measurement():
    buffer = make_buffer()
    buffer.submit(1, np.array([1.0]), delay_steps=5)
    buffer.submit(2, np.array([2.0]), delay_steps=0)
    assert buffer.receive(2)[0][0] == 2
    assert buffer.receive(6)[0][0] == 2


def test_dropout_holds_last_packet_and_age_grows():
    buffer = make_buffer()
    buffer.config = {**buffer.config, "packet_loss_probability": 1.0}
    for i in range(1, 10):
        buffer.submit(i, np.array([float(i)]))
    measurement, age = buffer.receive(9)
    assert age == 9 and measurement[0] == 0 and buffer.dropped == 9

