using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModel
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
        [LoadColumn(4)]
        public float Measurement;
        /// <summary>
        /// Плотость.
        /// </summary>
        [LoadColumn(7)]
        public float Density;
        /// <summary>
        /// Термометрия, контроль.
        /// </summary>
        [LoadColumn(3), ColumnName("Label1")]
        public float ThermometryVerified;
        /// <summary>
        /// Сухость пара.
        /// </summary>
        [LoadColumn(5), ColumnName("Label2")]
        public float SteamDryness;
        /// <summary>
        /// Удельная энтальпия.
        /// </summary>
        [LoadColumn(6), ColumnName("Label3")]
        public float Enthalpy;
    }
}
