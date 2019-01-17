# !/usr/bin/python
# -*- coding: utf-8 -*-
import NTM


a = NTM.ntm()
#a.run()
a.loadBaseData()
a.fitNN()
a.predict()
a.weight2storyline()

stop = 0