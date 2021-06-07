﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SecondMLModel
{
    class OutputModelForMeasurement
    {
        /// <summary>
        /// Расходометрия начальная.
        /// </summary>
        [ColumnName("Score")]
        public float MeasurementInitial;
    }
}
