using Microsoft.ML;
using Microsoft.ML.Data;
using SecondMLModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BuildModelsConsole
{
    class BuildSecondMLModel
    {
        private static string TRAIN_DATA_FILEPATH = @"E:\PCO2\Данные для обучения\2 модель\*";
        private static string MODEL_FILE = @"E:\PCO2\Models\";
        private static MLContext mlContext = new MLContext(seed: 1);
        private static DataOperationsCatalog.TrainTestData Data;

        public static void CreateModels()
        {
            CreateData();
            CreateModelForBarometryInitial();
            CreateModelForThermometryInitial();
            CreateModelForMeasurementInitial();
            CreateModelForDensityInitial();
        }

        private static void CreateModelForBarometryInitial()
        {
            mlContext = new MLContext(seed: 1);
            var trainingPipeline = BuildPipelineForBarometryInitial();
            var mlModel = TrainModel(mlContext, Data.TrainSet, trainingPipeline);
            Evaluate(mlModel, Data.TrainSet, "Label1");
            SaveModel(mlContext, mlModel, MODEL_FILE + "BarometryInitial.zip", Data.TrainSet.Schema);
        }
        private static void CreateModelForThermometryInitial()
        {
            mlContext = new MLContext(seed: 1);
            var trainingPipeline = BuildPipelineForThermometryInitial();
            var mlModel = TrainModel(mlContext, Data.TrainSet, trainingPipeline);
            Evaluate(mlModel, Data.TrainSet, "Label2");
            SaveModel(mlContext, mlModel, MODEL_FILE + "ThermometryInitial.zip", Data.TrainSet.Schema);
        }
        private static void CreateModelForMeasurementInitial()
        {
            mlContext = new MLContext(seed: 1);
            var trainingPipeline = BuildPipelineForMeasurementInitial();
            var mlModel = TrainModel(mlContext, Data.TrainSet, trainingPipeline);
            Evaluate(mlModel, Data.TrainSet, "Label3");
            SaveModel(mlContext, mlModel, MODEL_FILE + "MeasurementInitial.zip", Data.TrainSet.Schema);
        }
        private static void CreateModelForDensityInitial()
        {
            mlContext = new MLContext(seed: 1);
            var trainingPipeline = BuildPipelineForDensityInitial();
            var mlModel = TrainModel(mlContext, Data.TrainSet, trainingPipeline);
            Evaluate(mlModel, Data.TrainSet, "Label4");
            SaveModel(mlContext, mlModel, MODEL_FILE + "DensityInitial.zip", Data.TrainSet.Schema);
        }

        private static IEstimator<ITransformer> BuildPipelineForBarometryInitial()
        {
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Depth", "Barometry", "Thermometry", "Measurement", "Density" });
            var trainer = mlContext.Regression.Trainers.LightGbm("Label1", numberOfIterations: 700, numberOfLeaves: 60, minimumExampleCountPerLeaf: 2);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }
        private static IEstimator<ITransformer> BuildPipelineForThermometryInitial()
        {
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Depth", "Barometry", "Thermometry", "Measurement", "Density" });
            var trainer = mlContext.Regression.Trainers.FastTree("Label2", numberOfTrees: 600, numberOfLeaves: 60, minimumExampleCountPerLeaf: 5);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }
        private static IEstimator<ITransformer> BuildPipelineForMeasurementInitial()
        {
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Depth", "Barometry", "Thermometry", "Measurement", "Density" });
            var trainer = mlContext.Regression.Trainers.LightGbm("Label3", numberOfIterations: 600, numberOfLeaves: 65, minimumExampleCountPerLeaf: 2);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }
        private static IEstimator<ITransformer> BuildPipelineForDensityInitial()
        {
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Depth", "Barometry", "Thermometry", "Measurement", "Density" });
            var trainer = mlContext.Regression.Trainers.LightGbm("Label4", numberOfIterations: 600, numberOfLeaves: 65, minimumExampleCountPerLeaf: 2);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline;
        }

        private static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            return model;
        }

        private static void Evaluate(ITransformer mlmodel, IDataView trainingDataView, string label)
        {
            Console.WriteLine($"=============== to get model's accuracy metrics {label} ===============");
            var data = mlmodel.Transform(trainingDataView);
            var metrics = mlContext.Regression.Evaluate(data, labelColumnName: label);
            PrintRegressionMetrics(metrics);
        }
        private static void CrossEvaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline, string label)
        {
            Console.WriteLine($"=============== Cross-validating to get model's accuracy metrics {label} ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: label);
            PrintRegressionFoldsAverageMetrics(crossValidationResults);

        }

        private static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
        private static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:f3}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:f4}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:f3}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:f3}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:f3}");
            Console.WriteLine($"*************************************************");
        }

        /// <summary>
        /// Сохранить модель в ФС.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="mlModel"></param>
        /// <param name="path">Путь по которому сохранять модлеь.</param>
        /// <param name="modelInputSchema"></param>
        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string path, DataViewSchema modelInputSchema)
        {
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, path);
            Console.WriteLine("The model is saved to {0}", path);
        }

        /// <summary>
        /// Подготовить данные для обучения.
        /// </summary>
        private static void CreateData()
        {
            //Загрузка данных из файлов.
            var data = mlContext.Data.LoadFromTextFile<InputModel>(TRAIN_DATA_FILEPATH, ';', true);

            //Удаляем пустые строки и строки, в которых отрицательные значения (-999,250; -9999).
            data = mlContext.Data.FilterRowsByMissingValues(data, new[] { "Depth", "Barometry", "Thermometry", "Measurement", "Density", "Label1", "Label2", "Label3", "Label4" });
            data = mlContext.Data.FilterByCustomPredicate<InputModel>(data, (t) => { return t.Barometry < 0 || t.Density < 0 || t.Depth < 0 || t.Measurement < 0 || t.Thermometry < 0 || t.MeasurementInitial < 0 || t.ThermometryInitial < 0 || t.BarometryInitial < 0 || t.DensityInitial < 0; });

            //Разделяем данные на тестовые и тренирововчные.
            Data = mlContext.Data.TrainTestSplit(data, 0.2, seed: 1);
        }
    }
}
