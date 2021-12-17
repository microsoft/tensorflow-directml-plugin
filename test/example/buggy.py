import absl.testing.absltest as absltest
import time

class BuggyTest1(absltest.TestCase):
    def testCase1(self):
        self.assertFalse(True)

    def testCase2(self):
        self.skipTest("lada")

class BuggyTest2(absltest.TestCase):
    def testCase1(self):
        self.assertFalse(False)

    def testBravo(self):
        self.assertFalse(True)

    def testCharlie(self):
        self.assertFalse(True)

if __name__ == "__main__":
    absltest.main()