using Microsoft.ML;
using Microsoft.ML.Data;

MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

Console.WriteLine("Loading data...");

IDataView trainData = mlContext.Data.LoadFromTextFile<ModelInput>(
    "train.tsv",
    separatorChar: '\t',
    hasHeader: false
);

IDataView testData = mlContext.Data.LoadFromTextFile<ModelInput>(
    "test.tsv",
    separatorChar: '\t',
    hasHeader: false
);

/** MODEL TRAINING ****************************************************************************/
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                          .Append(mlContext.Transforms.Concatenate("Feature", "Sentence", "Description"))
                          .Append(mlContext.Transforms.Text.TokenizeIntoWords("Feature"))
                          .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Feature"))
                          .Append(mlContext.Transforms.Conversion.MapValueToKey("Feature"))
                          .Append(mlContext.Transforms.Text.ProduceNgrams("Feature"))
                          .Append(mlContext.Transforms.NormalizeLpNorm("Feature"))
                          .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                              mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(featureColumnName: "Feature",
                              l2Regularization: 0, l1Regularization: 0))
                          .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

Console.WriteLine("Training model...");
ITransformer model = pipeline.Fit(trainData);
Console.WriteLine("Evaluating model performance...");

IDataView transformedTest = model.Transform(testData);
MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

// Display Metrics
Console.WriteLine();
Console.WriteLine($"Average Macro Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine("Log Loss per class:");
for (int i = 0; i < 4; i++)
{
    Console.WriteLine($"Class - {Enum.GetValues<ArticleIntents>()[i]}, Log Loss for class - {metrics.PerClassLogLoss[i]}");
}

Console.WriteLine();
Console.WriteLine("Classes:");
foreach (ArticleIntents value in Enum.GetValues<ArticleIntents>())
{
    Console.WriteLine($"{((int)value) - 1}: {value}");
}
Console.WriteLine();

Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

/** PREDICTION GENERATION *********************************************************************/
Console.WriteLine("Creating prediction engine...");

PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Ready to generate predictions.");

string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What do you want to say ? (Type Q to Quit)");
    input = Console.ReadLine()!;
    if(input.ToLowerInvariant() == "q" || string.IsNullOrWhiteSpace(input))
    {
        break;
    }
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);

    Console.WriteLine($"Matched intent - {(ArticleIntents)result.PredictedLabel}");
    Console.WriteLine();
}
while (true);

Console.WriteLine("Bye!");