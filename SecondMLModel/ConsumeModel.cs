using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SecondMLModel
{
    public class ConsumeModel
    {
        private PredictionEngine<InputModel, OutputModelForBarometry> PredictionEngineForBarometryInitial;
        private PredictionEngine<InputModel, OutputModelForThermometry> PredictionEngineForThermometryInitial;
        private PredictionEngine<InputModel, OutputModelForMeasurement> PredictionEngineForMeasurementInitial;
        private PredictionEngine<InputModel, OutputModelForDensity> PredictionEngineForDensityInitial;

        private static string MLNetModelPath = @"E:\PCO2\Models\";

        public ConsumeModel()
        {
            PredictionEngineForBarometryInitial = CreatePredictionEngine<OutputModelForBarometry>(MLNetModelPath + "BarometryInitial.zip");
            PredictionEngineForThermometryInitial = CreatePredictionEngine<OutputModelForThermometry>(MLNetModelPath + "ThermometryInitial.zip");
            PredictionEngineForMeasurementInitial = CreatePredictionEngine<OutputModelForMeasurement>(MLNetModelPath + "MeasurementInitial.zip");
            PredictionEngineForDensityInitial = CreatePredictionEngine<OutputModelForDensity>(MLNetModelPath + "DensityInitial.zip");
        }

        public OutputModel Predict(InputModel input)
        {
            OutputModelForBarometry resultForBarometryInitial = PredictionEngineForBarometryInitial.Predict(input);
            OutputModelForThermometry resultForThermometryInitial = PredictionEngineForThermometryInitial.Predict(input);
            OutputModelForMeasurement resultForMeasurementInitial = PredictionEngineForMeasurementInitial.Predict(input);
            OutputModelForDensity resultForDensityInitial = PredictionEngineForDensityInitial.Predict(input);

            OutputModel result = new OutputModel
            {
                BarometryInitial = resultForBarometryInitial.BarometryInitial,
                ThermometryInitial = resultForThermometryInitial.ThermometryInitial,
                MeasurementInitial = resultForMeasurementInitial.MeasurementInitial,
                DensityInitial = resultForDensityInitial.DensityInitial
            };
            return result;
        }

        private static PredictionEngine<InputModel, T> CreatePredictionEngine<T>(string path) where T : class, new()
        {
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load(path, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<InputModel, T>(mlModel);

            return predEngine;
        }
    }
}
