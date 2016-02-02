#!/usr/bin/env python

from unittest import TestCase, main
import numpy as np
import datetime
from bcp.ethoscan import (parse_ethoscan_line, parse_ethoscan_report,
                          align_ethoscan_data)


class TestEthoscan(TestCase):
    '''Test Ethoscan parameters are correctly calculated.'''

    def setUp(self):
        self.ethoscan_report_lines_1 = [\
        'EthoScan: Data for cage 1, from file 070815 Infection.exp.\r\n',
        'Analyzed from 7/8/2015 10:25:47 to 7/16/2015 15:56:45\r\n',
        '*****************************************************************************************************************\r\n',
        '\r\n',
        'Behavior codes (not all may be present in this recording):\r\n',
        'EFODA,Interaction with food hopper A (significant uptake found)\r\n',
        'TFODA,Interaction with food hopper A (no significant uptake)\r\n',
        'DWATR,Interaction with water dispenser (significant uptake found)\r\n',
        'TWATR,Interaction with water dispenser (no significant uptake)\r\n',
        'WHEEL,Interaction with wheel (>= 1 revolution)\r\n',
        'IHOME,Entered habitat (stable mass reading)\r\n',
        'THOME,Interaction with habitat (no stable mass reading)\r\n',
        'LLNGE,Long lounge (> 60 sec, no non-XY sensor interactions)\r\n',
        'SLNGE,Short lounge (5 - 60 sec, no non-XY sensor interactions)\r\n',
        'EFODB,Interaction with food hopper B (significant uptake found)\r\n',
        'TFODB,Interaction with food hopper B (no significant uptake)\r\n',
        '\r\n',
        'Time Budget (total = 197.5 hours), Animal #1\r\n',
        '(Time before first and after last behavior are not included in the total)\r\n',
        'Behav,Minutes,Percent\r\n',
        'efoda,096.4,00.81\r\n',
        'tfoda,389.1,03.28\r\n',
        'dwatr,013.5,00.11\r\n',
        'twatr,008.2,00.07\r\n',
        'wheel,4064.3,34.30\r\n',
        'ihome,5764.5,48.64\r\n',
        'thome,006.8,00.06\r\n',
        'llnge,987.9,08.34\r\n',
        'slnge,520.3,04.39\r\n',
        'efodb,000.0,00.00\r\n',
        'tfodb,000.0,00.00\r\n',
        '\r\n',
        'Non-wheel XY Ambulatory Budget (total = 511.7 meters), Animal #1\r\n',
        'Behav,Meters,Percent\r\n',
        'efoda,14.6,02.85\r\n',
        'tfoda,45.4,08.88\r\n',
        'dwatr,07.8,01.53\r\n',
        'twatr,05.9,01.15\r\n',
        'ihome,00.0,00.00\r\n',
        'thome,00.1,00.02\r\n',
        'llnge,194.1,37.94\r\n',
        'slnge,243.7,47.63\r\n',
        'efodb,00.0,00.00\r\n',
        'tfodb,00.0,00.00\r\n',
        '\r\n',
        'With-wheel XY Ambulatory Budget (wheel = 92590.0 meters), Animal #1\r\n',
        '(Wheel diameter of 11.43 cm assumed; scale if necessary)\r\n',
        'Behav,Meters,Percent\r\n',
        'efoda,14.6,00.02\r\n',
        'tfoda,45.4,00.05\r\n',
        'dwatr,07.8,00.01\r\n',
        'twatr,05.9,00.01\r\n',
        'wheel,92590.0,99.45\r\n',
        'ihome,00.0,00.00\r\n',
        'thome,00.1,00.00\r\n',
        'llnge,194.1,00.21\r\n',
        'slnge,243.7,00.26\r\n',
        'efodb,00.0,00.00\r\n',
        'tfodb,00.0,00.00\r\n',
        '\r\n',
        'Transition matrix, animal #1\r\n',
        ',efoda,tfoda,dwatr,twatr,wheel,ihome,thome,llnge,slnge,efodb,tfodb\r\n',
        'efoda,00.00,00.00,00.00,00.65,14.19,00.00,00.00,18.71,66.45,00.00,00.00,sum:100%\r\n',
        'tfoda,00.00,00.00,01.22,00.97,06.81,00.00,00.00,14.11,76.89,00.00,00.00,sum:100%\r\n',
        'dwatr,00.00,06.25,00.00,00.00,00.00,00.00,00.00,15.63,78.13,00.00,00.00,sum:100%\r\n',
        'twatr,06.06,03.03,00.00,00.00,03.03,03.03,00.00,15.15,69.70,00.00,00.00,sum:100%\r\n',
        'wheel,02.06,03.04,00.00,00.00,00.00,00.00,00.00,09.62,85.28,00.00,00.00,sum:100%\r\n',
        'ihome,00.00,00.00,00.00,00.00,00.00,00.00,00.00,09.05,90.95,00.00,00.00,sum:100%\r\n',
        'thome,00.00,00.00,00.00,00.00,00.00,00.00,00.00,03.45,96.55,00.00,00.00,sum:100%\r\n',
        'llnge,04.59,16.51,03.67,03.21,44.95,25.23,01.83,00.00,00.00,00.00,00.00,sum:100%\r\n',
        'slnge,07.66,21.41,01.19,01.32,54.61,10.42,03.39,00.00,00.00,00.00,00.00,sum:100%\r\n',
        'efodb,,,,,,,,,,,,[not found]\r\n',
        'tfodb,,,,,,,,,,,,[not found]\r\n',
        'The matrix is viewed left to right (not vertically). The first column shows the initial behaviors; the\r\n',
        'other columns along a given row show the percent probability of the first subsequent behavior.\r\n',
        '\r\n',
        "Behavior list follows. 'Amount' is cm (SLNGE, LLNGE), revolutions (WHEEL), grams (EFODx, IHOME) or mL (DWATR)\r\n",
        ' Sample,Start_Date,Start_Time,End_Time,Durat_Sec,Activity,Amount,Rear%,X_cm,Y_cm,S_cm\r\n',
        ' 000001,7/8/2015\t10:25:48,14:43:41,15474,LLNGE,126,00.0,10.2,14.6,126\r\n',
        ' 015475,7/8/2015\t14:43:42,18:29:43,13562,IHOME,13.315,00.0,8.0,15.3,000\r\n',
        ' 029037,7/8/2015\t18:29:44,18:35:40,357,LLNGE,210,04.2,9.3,24.1,210\r\n',
        ' 029394,7/8/2015\t18:35:41,18:35:47,7,TWATR,0,85.7,5.8,27.5,004\r\n',
        ' 029401,7/8/2015\t18:35:48,18:42:13,386,LLNGE,125,08.0,8.5,24.0,125\r\n',
        ' 029787,7/8/2015\t18:42:14,18:42:16,3,DWATR,0.033,33.3,6.3,20.5,000\r\n',
        ' 029790,7/8/2015\t18:42:17,18:42:23,7,SLNGE,013,71.4,6.6,16.2,013\r\n',
        ' 029797,7/8/2015\t18:42:24,18:43:01,38,TFODA,0,68.4,10.0,18.7,000\r\n',
        ' 029835,7/8/2015\t18:43:02,18:43:06,5,SLNGE,007,80.0,8.8,17.4,007\r\n',
        ' 029840,7/8/2015\t18:43:07,18:44:25,79,IHOME,13.369,00.0,7.8,13.5,000\r\n',
        ' 029919,7/8/2015\t18:44:26,18:44:45,20,SLNGE,020,00.0,8.0,19.2,020\r\n',
        ' 029939,7/8/2015\t18:44:46,18:45:39,54,DWATR,0.024,48.1,5.9,25.7,037\r\n',
        ' 029993,7/8/2015\t18:45:40,18:45:50,11,SLNGE,019,81.8,7.6,28.3,019\r\n',
        ' 030004,7/8/2015\t18:45:51,18:45:51,1,TFODA,0,100.0,8.5,20.3,000\r\n',
        ' 030005,7/8/2015\t18:45:52,19:21:38,2147,LLNGE,250,01.4,12.1,19.6,250\r\n',
        ' 032152,7/8/2015\t19:21:39,19:21:52,14,WHEEL,007,00.0,9.0,9.0,000\r\n',
        ' 032166,7/8/2015\t19:21:53,19:22:02,10,SLNGE,000,00.0,9.0,9.0,000\r\n',
        ' 032176,7/8/2015\t19:22:03,19:22:04,2,TFODA,0,100.0,8.8,11.4,006\r\n',
        ' 032178,7/8/2015\t19:22:05,19:22:31,27,SLNGE,043,40.7,9.6,11.7,043\r\n',
        ' 032205,7/8/2015\t19:22:32,19:27:37,306,IHOME,13.368,00.0,8.3,14.0,000\r\n',
        ' 032511,7/8/2015\t19:27:38,19:28:23,46,SLNGE,061,26.1,8.4,16.8,061\r\n',
        ' 032557,7/8/2015\t19:28:24,19:28:26,3,TFODA,0,66.7,10.3,28.8,007\r\n']

    def test_parse_ethoscan_line(self):
        # input line
        # ' 032557,7/8/2015\t19:28:24,19:28:26,3,TFODA,0,66.7,10.3,28.8,007\r\n'
        exp = ['032557', 'TFODA', '3', '0', '66.7', '10.3', '28.8', '007']
        obs = parse_ethoscan_line(self.ethoscan_report_lines_1[-1])
        self.assertEqual(obs, exp)

    def test_parse_ethoscan_report(self):
        # Test without a start_time.
        exp = np.array([
            [1.00000000e+00, 7.00000000e+00, 1.54740000e+04, 1.26000000e+02, 0.00000000e+00, 1.02000000e+01, 1.46000000e+01, 1.26000000e+02],
            [1.54750000e+04, 5.00000000e+00, 1.35620000e+04, 1.33150000e+01, 0.00000000e+00, 8.00000000e+00, 1.53000000e+01, 0.00000000e+00],
            [2.90370000e+04, 7.00000000e+00, 3.57000000e+02, 2.10000000e+02, 4.20000000e+00, 9.30000000e+00, 2.41000000e+01, 2.10000000e+02],
            [2.93940000e+04, 3.00000000e+00, 7.00000000e+00, 0.00000000e+00, 8.57000000e+01, 5.80000000e+00, 2.75000000e+01, 4.00000000e+00],
            [2.94010000e+04, 7.00000000e+00, 3.86000000e+02, 1.25000000e+02, 8.00000000e+00, 8.50000000e+00, 2.40000000e+01, 1.25000000e+02],
            [2.97870000e+04, 2.00000000e+00, 3.00000000e+00, 3.30000000e-02, 3.33000000e+01, 6.30000000e+00, 2.05000000e+01, 0.00000000e+00],
            [2.97900000e+04, 8.00000000e+00, 7.00000000e+00, 1.30000000e+01, 7.14000000e+01, 6.60000000e+00, 1.62000000e+01, 1.30000000e+01],
            [2.97970000e+04, 1.00000000e+00, 3.80000000e+01, 0.00000000e+00, 6.84000000e+01, 1.00000000e+01, 1.87000000e+01, 0.00000000e+00],
            [2.98350000e+04, 8.00000000e+00, 5.00000000e+00, 7.00000000e+00, 8.00000000e+01, 8.80000000e+00, 1.74000000e+01, 7.00000000e+00],
            [2.98400000e+04, 5.00000000e+00, 7.90000000e+01, 1.33690000e+01, 0.00000000e+00, 7.80000000e+00, 1.35000000e+01, 0.00000000e+00],
            [2.99190000e+04, 8.00000000e+00, 2.00000000e+01, 2.00000000e+01, 0.00000000e+00, 8.00000000e+00, 1.92000000e+01, 2.00000000e+01],
            [2.99390000e+04, 2.00000000e+00, 5.40000000e+01, 2.40000000e-02, 4.81000000e+01, 5.90000000e+00, 2.57000000e+01, 3.70000000e+01],
            [2.99930000e+04, 8.00000000e+00, 1.10000000e+01, 1.90000000e+01, 8.18000000e+01, 7.60000000e+00, 2.83000000e+01, 1.90000000e+01],
            [3.00040000e+04, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+02, 8.50000000e+00, 2.03000000e+01, 0.00000000e+00],
            [3.00050000e+04, 7.00000000e+00, 2.14700000e+03, 2.50000000e+02, 1.40000000e+00, 1.21000000e+01, 1.96000000e+01, 2.50000000e+02],
            [3.21520000e+04, 4.00000000e+00, 1.40000000e+01, 7.00000000e+00, 0.00000000e+00, 9.00000000e+00, 9.00000000e+00, 0.00000000e+00],
            [3.21660000e+04, 8.00000000e+00, 1.00000000e+01, 0.00000000e+00, 0.00000000e+00, 9.00000000e+00, 9.00000000e+00, 0.00000000e+00],
            [3.21760000e+04, 1.00000000e+00, 2.00000000e+00, 0.00000000e+00, 1.00000000e+02, 8.80000000e+00, 1.14000000e+01, 6.00000000e+00],
            [3.21780000e+04, 8.00000000e+00, 2.70000000e+01, 4.30000000e+01, 4.07000000e+01, 9.60000000e+00, 1.17000000e+01, 4.30000000e+01],
            [3.22050000e+04, 5.00000000e+00, 3.06000000e+02, 1.33680000e+01, 0.00000000e+00, 8.30000000e+00, 1.40000000e+01, 0.00000000e+00],
            [3.25110000e+04, 8.00000000e+00, 4.60000000e+01, 6.10000000e+01, 2.61000000e+01, 8.40000000e+00, 1.68000000e+01, 6.10000000e+01],
            [3.25570000e+04, 1.00000000e+00, 3.00000000e+00, 0.00000000e+00, 6.67000000e+01, 1.03000000e+01, 2.88000000e+01, 7.00000000e+00]])
        obs = parse_ethoscan_report(self.ethoscan_report_lines_1)
        np.testing.assert_array_equal(obs, exp)
        # Test with a start time.
        start_time = 3435.
        obs = parse_ethoscan_report(self.ethoscan_report_lines_1, start_time)
        exp[:, 0] += start_time
        np.testing.assert_array_equal(obs, exp)

    def test_align_ethoscan_data(self):
        # Simulate a situation where 1 day has elapsed since the beginning of
        # the experiment and the beginning of the Ethoscan. The Ethoscan will 
        # report will be for 1h of activity.
        exp_start = datetime.datetime(2015, 1, 1, 6, 0, 0)
        eth_start = datetime.datetime(2015, 1, 2, 6, 0, 0)
        # Assume that the experiment has been stopped for 1h to collect samples
        # from the mice. 
        times = np.concatenate((np.arange(23 * 3600),
                                np.arange(3600) + 24 * 3600))
        # Mock Ethoscan behavioral classification.
        edata = np.array([[1, 3, 9, 0, 0, 15.25, 13.5, 0],
                          [10, 4, 3000, 1560, 0, 2.5, 2.5, 0],
                          [3010, 0, 570, .6, 0, 14.5, 14.5, 0],
                          [3580, 6, 19, 0, 13, 16.5, 7.5, 300]])
        # The times post experiment start that our Ethoscan observations occur
        # at are different than the actual indices in the times vector due to
        # the experiment being paused. The actual index of these observations
        # will be shifted back by 3600 (3600 observations in 1h).
        times_exp = np.array([[86401, 86410],
                              [86410, 89410],
                              [89410, 89980],
                              [89980, 89999]])
        exp = times_exp - 3600
        # Generate observed data.
        obs = np.empty((4,2))
        for i, e in enumerate(edata):
            obs[i] = align_ethoscan_data(exp_start, eth_start, e, times)

        np.testing.assert_array_equal(obs, exp)

        # Simulate a situation where 2 pauses have occurred in experimental
        # recording (i.e. 2 days of sampling). Seconds between experiment start
        # and experiment end: 487671.
        exp_start = datetime.datetime(2015, 2, 28, 7, 42, 15)
        exp_end = datetime.datetime(2015, 3, 5, 23, 10, 6)
        # The Ethoscan will start with 97200 seconds to go.
        eth_start = datetime.datetime(2015, 3, 4, 20, 10, 6)
        # Our mock times data has a loss of 1000 samples (1000 seconds where 
        # Promethion machine was off) at the first break, and 2000 samples at
        # the second break. 
        times = np.concatenate((np.arange(79000),
                                80000 + np.arange(115425),
                                197425 + np.arange(290246)))
        # Mock Ethoscan behavioral classifications.
        edata = np.array([[1, 3, 9, 0, 0, 15.25, 13.5, 0],
                          [10, 4, 3000, 1560, 0, 2.5, 2.5, 0],
                          [3010, 0, 570, .6, 0, 14.5, 14.5, 0],
                          [3580, 6, 19, 0, 13, 16.5, 7.5, 300],
                          [3599, 8, 93600, 0, 15.25, 2.5, 100]])

        times_exp = np.array([[390472, 390481],
                              [390481, 393481],
                              [393481, 394051],
                              [394051, 394070],
                              [394070, 487670]])
        exp = times_exp - 3000
        # Generate observed data.
        obs = np.empty((5,2))
        for i, e in enumerate(edata):
            obs[i] = align_ethoscan_data(exp_start, eth_start, e, times)

        np.testing.assert_array_equal(obs, exp)

# run unit tests if run from command-line
if __name__ == '__main__':
    main()
