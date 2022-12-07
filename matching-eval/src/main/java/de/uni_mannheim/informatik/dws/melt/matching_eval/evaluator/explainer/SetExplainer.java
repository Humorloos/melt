package de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.explainer;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.TextExtractorSet;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.TextMoleculeExtractorImpl;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.rdf.model.Resource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class SetExplainer implements IExplainerResourceWithJenaOntology {

    private final TextExtractorSet extractor = new TextExtractorSet();
    private OntModel ontModel;

    @Override
    public Map<String, String> getResourceFeatures(String uri) {
        Resource resource = ontModel.getResource(uri);
        String setString = String.join("|", extractor.extract(resource));
        return Map.of("SET", setString);
    }

    @Override
    public List<String> getResourceFeatureNames() {
        return List.of("SET");
    }

    @Override
    public void setOntModel(OntModel ontModel) {
        this.ontModel = ontModel;
    }
}
