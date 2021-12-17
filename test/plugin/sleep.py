import absl.testing.absltest as absltest
import time

class VisibleDevicesTest(absltest.TestCase):
    def test(self):
        time.sleep(10)

if __name__ == "__main__":
    absltest.main()