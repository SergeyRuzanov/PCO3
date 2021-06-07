using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SecondMLModel
{
    class InputModel
    {
        /// <summary>
        /// Глубина.
        /// </summary>
        [LoadColumn(0)]
        public float Depth;
        /// <summary>
        /// Барометрия
        /// </summary>
        [LoadColumn(1)]
        public float Barometry;
        /// <summary>
        /// Термометрия.
        /// </summary>
        [LoadColumn(2)]
        public float Thermometry;
        /// <summary>
        /// Расходометрия.
        /// </summary>
        [LoadColumn(3)]
        public float Measurement;
        /// <summary>
        /// Плотость.
        /// </summary>
        [LoadColumn(4)]
        public float Density;
        /// <summary>
        /// Барометрия начальная.
        /// </summary>
        [LoadColumn(5), ColumnName("Label1")]
        public float BarometryInitial;
        /// <summary>
        /// Термометрия начальная.
        /// </summary>
        [LoadColumn(6), ColumnName("Label2")]
        public float ThermometryInitial;
        /// <summary>
        /// Расходометрия начальная.
        /// </summary>
        [LoadColumn(7), ColumnName("Label3")]
        public float MeasurementInitial;
        /// <summary>
        /// Плотость начальная.
        /// </summary>
        [LoadColumn(8), ColumnName("Label4")]
        public float DensityInitial;
    }
}
