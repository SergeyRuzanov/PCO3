using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModel
{
    class OutputModelForThermometryVerified
    {
        /// <summary>
        /// Термометрия, контроль.
        /// </summary>
        [ColumnName("Score")]
        public float ThermometryVerified;
    }
}
