using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModel
{
    public class ConsumeModel
    {
        private PredictionEngine<InputModel, OutputModelForThermometryVerified> PredictionEngineForThermometryVerified;
        private PredictionEngine<InputModel, OutputModelForSteamDryness> PredictionEngineForSteamDryness;
        private PredictionEngine<InputModel, OutputModelForEnthalpy> PredictionEngineForEnthalpy;

        private static string MLNetModelPath = @"E:\PCO2\Models\";

        public ConsumeModel()
        {
            PredictionEngineForThermometryVerified = CreatePredictionEngine<OutputModelForThermometryVerified>(MLNetModelPath + "ThermometryVerified.zip");
            PredictionEngineForSteamDryness = CreatePredictionEngine<OutputModelForSteamDryness>(MLNetModelPath + "SteamDryness.zip");
            PredictionEngineForEnthalpy = CreatePredictionEngine<OutputModelForEnthalpy>(MLNetModelPath + "Enthalpy.zip");
        }

        public OutputModel Predict(InputModel input)
        {
            OutputModelForThermometryVerified resultForThermometryVerified = PredictionEngineForThermometryVerified.Predict(input);
            OutputModelForSteamDryness resultForSteamDryness = PredictionEngineForSteamDryness.Predict(input);
            OutputModelForEnthalpy resultForEnthalpy = PredictionEngineForEnthalpy.Predict(input);

            OutputModel result = new OutputModel { ThermometryVerified = resultForThermometryVerified.ThermometryVerified, SteamDryness = resultForSteamDryness.SteamDryness, Enthalpy = resultForEnthalpy.Enthalpy };
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
