using Microsoft.ML.Data;

public class ModelInput
{
    public ModelInput(string utterance)
    {
        Sentence = utterance;
    }

    [LoadColumn(2)]
    [ColumnName(@"Description")]
    public string Description { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Sentence")]
    public string Sentence { get; set; }

    [LoadColumn(0)]
    [ColumnName(@"Label")]
    public float Label { get; set; }

}
