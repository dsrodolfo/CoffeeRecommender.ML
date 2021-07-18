using CoffeeRecommender.Domain.Entities;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace CoffeeRecommender.Domain.Services
{
    public class MLService
    {
        public IDataView LoadTrainingData(MLContext mLContext)
        {
            var trainingDataPath = Path.Combine(GetInfrastructureDirectory(), "Data", "recommendation-ratings-train.csv");
            IDataView trainingDataView = mLContext.Data.LoadFromTextFile<CoffeeRatingEntity>(trainingDataPath, hasHeader: true, separatorChar: ',');

            return trainingDataView;
        }

        public IDataView LoadTestData(MLContext mLContext)
        {
            var testDataPath = Path.Combine(GetInfrastructureDirectory(), "Data", "recommendation-ratings-test.csv");
            IDataView testDataView = mLContext.Data.LoadFromTextFile<CoffeeRatingEntity>(testDataPath, hasHeader: true, separatorChar: ',');

            return testDataView;
        }

        public ITransformer BuildAndTrainModel(MLContext mLContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "UserIdEnconded", inputColumnName: "UserId")
              .Append(mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "CoffeeIdEnconded", inputColumnName: "CoffeeId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEnconded",
                MatrixRowIndexColumnName = "CoffeeIdEnconded",
                LabelColumnName = "Rating",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mLContext.Recommendation().Trainers.MatrixFactorization(options));
            Console.WriteLine(">>>>>>> Training the Model <<<<<<<");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public void EvaluateModel(MLContext mLContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine(">>>>>>> Evaluating the Model <<<<<<<");
            var prediction = model.Transform(testDataView);
            var metrics = mLContext.Regression.Evaluate(prediction, labelColumnName: "Rating", scoreColumnName: "Score");

            Console.WriteLine($"Root Mean Squared Erros: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"RSquared {metrics.RSquared}");
        }

        private string GetInfrastructureDirectory() 
            => Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.GetDirectories()[3].ToString();
    }
}