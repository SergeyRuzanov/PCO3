using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BuildModelsConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            BuildFirstMLModel.CreateModels();
            BuildSecondMLModel.CreateModels();
            Console.ReadLine();
            //ConsumeModel consumeModel = new ConsumeModel();
            ///*while (true)
            //{
            //    try
            //    {
            //        float depth, barometry, thermometry, measurement, density;
            //        Console.Write("Глубина: ");
            //        depth = float.Parse(Console.ReadLine());
            //        Console.Write("Барометрия: ");
            //        barometry = float.Parse(Console.ReadLine());
            //        Console.Write("Термометрия: ");
            //        thermometry = float.Parse(Console.ReadLine());
            //        Console.Write("Расходометрия: ");
            //        measurement = float.Parse(Console.ReadLine());
            //        Console.Write("Плотность: ");
            //        density = float.Parse(Console.ReadLine());

            //        InputModel input = new InputModel()
            //        {
            //            Depth = depth,
            //            Barometry = barometry,
            //            Thermometry = thermometry,
            //            Measurement = measurement,
            //            Density = density
            //        };
            //        OutputModel prediction = consumeModel.Predict(input);
            //        Console.WriteLine("===============================================");
            //        Console.WriteLine($"Термометрия, контроль: {prediction.ThermometryVerified}");
            //        Console.WriteLine($"Степень сухости пара: {prediction.SteamDryness}");
            //        Console.WriteLine($"Удельная энтальпия: {prediction.Enthalpy}");
            //    }
            //    catch(Exception ex)
            //    {
            //        Console.WriteLine("Ошибка! Повторите ввод!");
            //    }
            //}*/
            //InputModel inputModel = new InputModel()
            //{
            //    Depth = 1317.13F,
            //    Barometry = 10.624F,
            //    Thermometry = 332.488F,
            //    Measurement = 1522.776F,
            //    Density = 0.37f
            //};
            //OutputModel predict = consumeModel.Predict(inputModel);
            //Console.WriteLine($"{predict.ThermometryVerified}   {predict.SteamDryness}   {predict.Enthalpy}");
            //inputModel = new InputModel()
            //{
            //    Depth = 1317.23F,
            //    Barometry = 10.624F,
            //    Thermometry = 332.368F,
            //    Measurement = 1519.888F,
            //    Density = 0.385f
            //};

            //predict = consumeModel.Predict(inputModel);
            //Console.WriteLine($"{predict.ThermometryVerified}   {predict.SteamDryness}   {predict.Enthalpy}");
            //Console.ReadKey();
        }
    }
}
