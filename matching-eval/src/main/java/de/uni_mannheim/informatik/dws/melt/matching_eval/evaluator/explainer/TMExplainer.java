package de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.explainer;

import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.util.textExtractors.kBert.TextMoleculeExtractorImpl;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.rdf.model.Resource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Explainer for making texts extracted by TextMoleculeExtractorImpl readable and analyzable
 */
public class TMExplainer implements IExplainerResourceWithJenaOntology {

    private final TextMoleculeExtractorImpl extractor = new TextMoleculeExtractorImpl(true, true, false);
    private OntModel ontModel;

    public TMExplainer() {
        extractor.setUseIndex(false);
    }

    @Override
    public Map<String, String> getResourceFeatures(String uri) {
        Resource resource = ontModel.getResource(uri);
        Map<String, Set<?>> textMolecule = extractor.moleculesFromResource(resource).stream().findFirst().orElse(null);
        List<String> objectStatementTexts = new ArrayList<>();
        List<String> subjectStatementTexts = new ArrayList<>();
        textMolecule.get("s").forEach(stmt -> {
            Map<String, String> mapStmt = (Map<String, String>) stmt;
            if (mapStmt.get("r").equals("s")) {
                subjectStatementTexts.add(mapStmt.get("n") + " " + mapStmt.get("p"));
            } else {
                objectStatementTexts.add(mapStmt.get("p") + ' ' + mapStmt.get("n"));
            }
        });
        String slmr = (subjectStatementTexts.isEmpty() ? "" : String.join("|", subjectStatementTexts) + " ")
                + "<" + textMolecule.get("t").stream()
                .map(target -> ((String) target).toUpperCase()).collect(Collectors.joining("|"))
                + ">"
                + (objectStatementTexts.isEmpty() ? "" : " " + String.join("|", objectStatementTexts));
        return Map.of("TM", slmr);
    }

    @Override
    public List<String> getResourceFeatureNames() {
        return List.of("TM");
    }

    @Override
    public void setOntModel(OntModel ontModel) {
        this.ontModel = ontModel;
    }
}
