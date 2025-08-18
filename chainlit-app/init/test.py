import os
from dotenv import load_dotenv
import numpy as np
from dashscope import TextEmbedding, Generation
# 导入InterSystems IRIS Python驱动
import iris as irisnative

load_dotenv()

tested_patient = """
{
    "resourceType": "Bundle",
    "id": "00a9474c-7a59-11f0-8244-0242ac140003",
    "type": "searchset",
    "timestamp": "2025-08-16T04:25:13Z",
    "total": 21,
    "link": [
        {
            "relation": "self",
            "url": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Patient/1594/$everything"
        }
    ],
    "entry": [
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Patient/1594",
            "resource": {
                "resourceType": "Patient",
                "id": "1594",
                "name": [
                    {
                        "use": "official",
                        "family": "张",
                        "given": [
                            "三"
                        ]
                    }
                ],
                "gender": "male",
                "birthDate": "1980-01-15",
                "address": [
                    {
                        "use": "home",
                        "line": [
                            "健康路88号"
                        ],
                        "city": "北京市",
                        "district": "海淀区",
                        "postalCode": "100080"
                    }
                ],
                "telecom": [
                    {
                        "system": "phone",
                        "value": "13800138000",
                        "use": "mobile"
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/1595",
            "resource": {
                "resourceType": "Encounter",
                "id": "1595",
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "IMP",
                    "display": "Inpatient"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/encounter-type",
                                "code": "ADMS",
                                "display": "Admission"
                            }
                        ]
                    }
                ],
                "subject": {
                    "reference": "Patient/1594"
                },
                "period": {
                    "start": "2023-03-10T08:30:00+08:00",
                    "end": "2023-03-20T10:00:00+08:00"
                },
                "reasonCode": [
                    {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "I10",
                                "display": "原发性高血压"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Observation/1596",
            "resource": {
                "resourceType": "Observation",
                "id": "1596",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "laboratory",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "2345-7",
                            "display": "胆固醇, 血清"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1595"
                },
                "effectiveDateTime": "2023-03-11T09:15:00+08:00",
                "valueQuantity": {
                    "value": 5.2,
                    "unit": "mmol/L",
                    "system": "http://unitsofmeasure.org",
                    "code": "mmol/L"
                },
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": "H",
                                "display": "High"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/DiagnosticReport/1597",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": "1597",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/diagnostic-service-sections",
                                "code": "RAD",
                                "display": "Radiology"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "36598-0",
                            "display": "胸部X线检查"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1595"
                },
                "effectiveDateTime": "2023-03-12T14:30:00+08:00",
                "issued": "2023-03-12T16:45:00+08:00",
                "conclusion": "双肺纹理略增粗，未见明显实变影",
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/1598",
            "resource": {
                "resourceType": "Condition",
                "id": "1598",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                            "display": "Confirmed"
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "I10",
                            "display": "原发性高血压"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1595"
                },
                "onsetDateTime": "2023-03-10T08:30:00+08:00",
                "recorder": {
                    "reference": "Practitioner/p1"
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/MedicationRequest/1599",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "1599",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "314049",
                            "display": "硝苯地平"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1595"
                },
                "authoredOn": "2023-03-10T10:15:00+08:00",
                "requester": {
                    "reference": "Practitioner/p1"
                },
                "dosageInstruction": [
                    {
                        "text": "每日一次，每次10mg，口服",
                        "timing": {
                            "repeat": {
                                "frequency": 1,
                                "period": 1,
                                "periodUnit": "d"
                            }
                        },
                        "route": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/routes",
                                    "code": "PO",
                                    "display": "Oral"
                                }
                            ]
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": 10,
                                    "unit": "mg",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg"
                                }
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/1600",
            "resource": {
                "resourceType": "Encounter",
                "id": "1600",
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "IMP",
                    "display": "Inpatient"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/encounter-type",
                                "code": "ADMS",
                                "display": "Admission"
                            }
                        ]
                    }
                ],
                "subject": {
                    "reference": "Patient/1594"
                },
                "period": {
                    "start": "2023-07-05T09:00:00+08:00",
                    "end": "2023-07-12T11:30:00+08:00"
                },
                "reasonCode": [
                    {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "E11.9",
                                "display": "2型糖尿病，无并发症"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Observation/1601",
            "resource": {
                "resourceType": "Observation",
                "id": "1601",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "laboratory",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "2345-7",
                            "display": "胆固醇, 血清"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1600"
                },
                "effectiveDateTime": "2023-07-06T08:45:00+08:00",
                "valueQuantity": {
                    "value": 7.8,
                    "unit": "mmol/L",
                    "system": "http://unitsofmeasure.org",
                    "code": "mmol/L"
                },
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": "H",
                                "display": "High"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/DiagnosticReport/1602",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": "1602",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/diagnostic-service-sections",
                                "code": "LAB",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "14647-2",
                            "display": "血糖检测"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1600"
                },
                "effectiveDateTime": "2023-07-07T10:20:00+08:00",
                "issued": "2023-07-07T11:30:00+08:00",
                "conclusion": "空腹血糖明显升高，符合糖尿病表现",
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/1603",
            "resource": {
                "resourceType": "Condition",
                "id": "1603",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                            "display": "Confirmed"
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "E11.9",
                            "display": "2型糖尿病，无并发症"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1600"
                },
                "onsetDateTime": "2023-07-05T09:00:00+08:00",
                "recorder": {
                    "reference": "Practitioner/p1"
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/MedicationRequest/1604",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "1604",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "6809",
                            "display": "二甲双胍"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1600"
                },
                "authoredOn": "2023-07-05T14:20:00+08:00",
                "requester": {
                    "reference": "Practitioner/p1"
                },
                "dosageInstruction": [
                    {
                        "text": "每日三次，每次500mg，口服，饭后服用",
                        "timing": {
                            "repeat": {
                                "frequency": 3,
                                "period": 1,
                                "periodUnit": "d"
                            }
                        },
                        "route": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/routes",
                                    "code": "PO",
                                    "display": "Oral"
                                }
                            ]
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": 500,
                                    "unit": "mg",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg"
                                }
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/1605",
            "resource": {
                "resourceType": "Encounter",
                "id": "1605",
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "Ambulatory"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/encounter-type",
                                "code": "ROUTINE",
                                "display": "Routine"
                            }
                        ]
                    }
                ],
                "subject": {
                    "reference": "Patient/1594"
                },
                "period": {
                    "start": "2023-04-15T14:00:00+08:00",
                    "end": "2023-04-15T14:45:00+08:00"
                },
                "reasonCode": [
                    {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "I10",
                                "display": "原发性高血压"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Observation/1606",
            "resource": {
                "resourceType": "Observation",
                "id": "1606",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "vital-signs",
                                "display": "Vital Signs"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8480-6",
                            "display": "收缩压"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1605"
                },
                "effectiveDateTime": "2023-04-15T14:10:00+08:00",
                "valueQuantity": {
                    "value": 145,
                    "unit": "mmHg",
                    "system": "http://unitsofmeasure.org",
                    "code": "mmHg"
                },
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": "H",
                                "display": "High"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/DiagnosticReport/1607",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": "1607",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/diagnostic-service-sections",
                                "code": "LAB",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "2345-7",
                            "display": "胆固醇, 血清"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1605"
                },
                "effectiveDateTime": "2023-04-15T14:30:00+08:00",
                "issued": "2023-04-15T14:40:00+08:00",
                "conclusion": "胆固醇水平较住院期间有所下降，但仍高于正常范围",
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/1608",
            "resource": {
                "resourceType": "Condition",
                "id": "1608",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                            "display": "Confirmed"
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "I10",
                            "display": "原发性高血压"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1605"
                },
                "onsetDateTime": "2023-04-15T14:00:00+08:00",
                "recorder": {
                    "reference": "Practitioner/p1"
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/MedicationRequest/1609",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "1609",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "314049",
                            "display": "硝苯地平"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1605"
                },
                "authoredOn": "2023-04-15T14:45:00+08:00",
                "requester": {
                    "reference": "Practitioner/p1"
                },
                "dosageInstruction": [
                    {
                        "text": "每日一次，每次10mg，口服",
                        "timing": {
                            "repeat": {
                                "frequency": 1,
                                "period": 1,
                                "periodUnit": "d"
                            }
                        },
                        "route": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/routes",
                                    "code": "PO",
                                    "display": "Oral"
                                }
                            ]
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": 10,
                                    "unit": "mg",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg"
                                }
                            }
                        ]
                    }
                ],
                "dispenseRequest": {
                    "quantity": {
                        "value": 30,
                        "unit": "片"
                    },
                    "expectedSupplyDuration": {
                        "value": 30,
                        "unit": "天",
                        "system": "http://unitsofmeasure.org",
                        "code": "d"
                    }
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/1610",
            "resource": {
                "resourceType": "Encounter",
                "id": "1610",
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "Ambulatory"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/encounter-type",
                                "code": "ROUTINE",
                                "display": "Routine"
                            }
                        ]
                    }
                ],
                "subject": {
                    "reference": "Patient/1594"
                },
                "period": {
                    "start": "2023-08-20T09:30:00+08:00",
                    "end": "2023-08-20T10:15:00+08:00"
                },
                "reasonCode": [
                    {
                        "coding": [
                            {
                                "system": "http://hl7.org/fhir/sid/icd-10",
                                "code": "E11.9",
                                "display": "2型糖尿病，无并发症"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Observation/1611",
            "resource": {
                "resourceType": "Observation",
                "id": "1611",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "laboratory",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "2345-7",
                            "display": "胆固醇, 血清"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1610"
                },
                "effectiveDateTime": "2023-08-20T09:40:00+08:00",
                "valueQuantity": {
                    "value": 6.5,
                    "unit": "mmol/L",
                    "system": "http://unitsofmeasure.org",
                    "code": "mmol/L"
                },
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": "H",
                                "display": "High"
                            }
                        ]
                    }
                ],
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/DiagnosticReport/1612",
            "resource": {
                "resourceType": "DiagnosticReport",
                "id": "1612",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/diagnostic-service-sections",
                                "code": "LAB",
                                "display": "Laboratory"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "14647-2",
                            "display": "血糖检测"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1610"
                },
                "effectiveDateTime": "2023-08-20T09:50:00+08:00",
                "issued": "2023-08-20T10:00:00+08:00",
                "conclusion": "血糖水平较住院期间有所改善，但仍需继续药物治疗和饮食控制",
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/1613",
            "resource": {
                "resourceType": "Condition",
                "id": "1613",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed",
                            "display": "Confirmed"
                        }
                    ]
                },
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis",
                                "display": "Encounter Diagnosis"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "E11.9",
                            "display": "2型糖尿病，无并发症"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1610"
                },
                "onsetDateTime": "2023-08-20T09:30:00+08:00",
                "recorder": {
                    "reference": "Practitioner/p1"
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/MedicationRequest/1614",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "1614",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "6809",
                            "display": "二甲双胍"
                        }
                    ]
                },
                "subject": {
                    "reference": "Patient/1594"
                },
                "encounter": {
                    "reference": "Encounter/1610"
                },
                "authoredOn": "2023-08-20T10:15:00+08:00",
                "requester": {
                    "reference": "Practitioner/p1"
                },
                "dosageInstruction": [
                    {
                        "text": "每日三次，每次500mg，口服，饭后服用",
                        "timing": {
                            "repeat": {
                                "frequency": 3,
                                "period": 1,
                                "periodUnit": "d"
                            }
                        },
                        "route": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/routes",
                                    "code": "PO",
                                    "display": "Oral"
                                }
                            ]
                        },
                        "doseAndRate": [
                            {
                                "doseQuantity": {
                                    "value": 500,
                                    "unit": "mg",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg"
                                }
                            }
                        ]
                    }
                ],
                "dispenseRequest": {
                    "quantity": {
                        "value": 90,
                        "unit": "片"
                    },
                    "expectedSupplyDuration": {
                        "value": 30,
                        "unit": "天",
                        "system": "http://unitsofmeasure.org",
                        "code": "d"
                    }
                },
                "meta": {
                    "lastUpdated": "2025-08-16T03:55:19Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        }
    ]
}
"""

class IRISRAG:
    def __init__(self, host="localhost", port=1980, namespace="MCP", username="superuser", password="SYS"):
        #print(os.getenv("DASHSCOPE_API_KEY"))
        """初始化IRIS连接"""
        self.documents = []  # 本地缓存文档（可选）
        
        self.connection = irisnative.createConnection(host, port, namespace, username, password)
        
        # 确保表结构存在
        self._setup_database()
    
    def _setup_database(self):
        """创建必要的表结构"""
        try:
            cursor = self.connection.cursor()
            # 创建文档表，包含ID、内容和嵌入向量
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS RagSystem.Document (
                    Content VARCHAR(10000),
                    Embedding VECTOR(FLOAT,1536)  -- Qwen Embedding向量维度
                )
                """
            )
            cursor.close()
        except Exception as e:
            print(f"数据库设置可能已存在或出错: {e}")
    
    def _get_embedding(self, texts):
        """获取文本的嵌入向量"""
        # DashScope Embedding API调用
        response = TextEmbedding.call(
            model="text-embedding-v2",  # Qwen3 Embedding专用模型
            input=texts,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        if response.status_code == 200:
            # 提取嵌入向量
            embeddings = [item["embedding"] for item in response.output["embeddings"]]
            return embeddings
        else:
            raise Exception(f"Embedding API error: {response.code}, {response.message}")
    
    def add_documents(self, documents):
        """添加文档到IRIS向量库"""
        # 获取所有文档的嵌入
        embeddings = self._get_embedding(documents)
        cursor = self.connection.cursor()
        # 清除文档
        cursor.execute(
            """
            truncate table RagSystem.Document
            """
        )
        print("清除文档")
        # 将文档和向量插入IRIS
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # 将向量转换为IRIS可接受的格式
            # IRIS中%Vector类型可以表示为逗号分隔的字符串
            embedding_str = ",".join(map(str, embedding))
            # 插入文档
            cursor.execute(
                """
                INSERT INTO RagSystem.Document (Content, Embedding)
                VALUES (?, ?)
                """,
                [doc, embedding_str]
            )
        cursor.close()
        print(f"成功添加 {len(documents)} 个文档到IRIS向量库")
        self.documents.extend(documents)  # 本地缓存（可选）
    
    def retrieve(self, query, top_k=5):
        """使用IRIS检索最相关的文档"""
        # 获取查询的嵌入
        query_embedding = self._get_embedding([query])[0]
        query_embedding_str = ",".join(map(str, query_embedding))
        #print(query_embedding_str)
        # 使用IRIS的向量相似度搜索功能
        # 假设IRIS已启用向量搜索功能，这里使用余弦相似度
        cursor = self.connection.cursor()
        sqlStr = """
            SELECT TOP ? Id, Content, VECTOR_DOT_PRODUCT(TO_VECTOR(?,float),Embedding) AS Similarity
            FROM DrugInfo.Insurance
            ORDER BY Similarity DESC
            where Similarity > 0.5
            """
        cursor.execute(
            sqlStr,
            [top_k,query_embedding_str]
        )
        results = cursor.fetchall()
        #print(results)
        cursor.close()
        # 处理结果
        retrieved_docs = []
        for row in results:
            retrieved_docs.append({
                "id": row[0],
                "content": row[1],
                "score": row[2]  # 相似度分数
            })
        
        return retrieved_docs
    
    def generate_answer(self, query, context_docs, patInfo):
        """使用Qwen-Plus生成答案"""
        # 构建prompt，包含上下文和问题
        #print(context_docs)
        context = "\n\n".join([doc["RuleInsurance"] for doc in context_docs])
        prompt = f"""基于以下上下文信息回答医保拒付风险相关的问题：

医保规则信息：{context}

问题：{query}

患者信息:{patInfo}

要依据获取到的医保规则仔细分析，确认报销约束是不是都得到了满足。
例如盐酸右美托咪定的报销条件为：成人术前镇静/抗焦虑，则只有患者为成人且有手术医嘱且术前有类似焦虑的诊断或并病程记录才算满足条件。
又如溴芬酸钠的报销条件为：限眼部手术后炎症。如果患者信息中没有眼部手术的记录或没有眼部手术后炎症的信息，则未满足报销条件。
如果上下文中没有明确的医保报销约束，则根据你的常识和患者的实际病情判断用药的适应症是否存在。如果患者病情中存在适应症，则没有医保拒付风险；如果患者病情中不存在适应症，则有医保拒付风险。
对于二线用药，如果患者病情中没有一线用药记录或没有一线用药无效的记录，也应判断为没有适应症。在解释部分应该加入一些解释，说明如果要用这种二线药物，对应的一线药物是什么。
只要有一个药物存在拒付风险，则整体来看就有拒付风险。
回答时要简洁，按顺序包括三个方面内容：
1. 结论：根据当前上下文中的信息明确回答是否有医保拒付风险，只回答有没有风险即可。
2. 规则：针对问题中的药物（而不是医保规则信息中的药物）逐条解释从上下文中获得的医保规则是什么样的。在说明医保规则时应严格复述上下文中记录的规则文本，不要创造内容。
3. 解释：针对问题中的每一种药物，如果有拒付风险，说明原因。如果没有拒付风险，说明支撑条件是什么。

"""
        #print(prompt)
        # 调用Qwen-Plus生成答案
        #print(prompt)
        response = Generation.call(
            model="qwen-plus",  # 使用Qwen-Plus模型
            prompt=prompt,
            temperature=0.3,
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        if response.status_code == 200:
            return response.output["text"]
        else:
            return f"生成答案时出错: {response.code}, {response.message}"
    
    def rag_query(self, query, top_k=2):
        """执行完整的RAG查询"""
        # 1. 检索相关文档
        relevant_docs = self.retrieve(query, top_k)
        print("aaa")
        print(relevant_docs)
        # 2. 生成最终答案
        answer = self.generate_answer(query, relevant_docs)
        
        return {
            "answer": answer,
            "sources": relevant_docs
        }

    def close(self):
        """关闭IRIS连接"""
        self.connection.close()

    def drug_query(self, query, top_k=2):
        """执行完整的RAG查询"""
        question = f"我想给患者开{query}，会有医保拒付的风险吗？"
        # 1. 检索相关文档
        relevant_docs = self.drug_retrieve(question, top_k)
        
        # 2. 生成最终答案
        answer = self.generate_answer(f"我想给患者开{query}，会有医保拒付的风险吗？", relevant_docs, tested_patient)
        
        return {
            "answer": answer,
            "sources": relevant_docs
        }
    
    def drug_retrieve(self, query, top_k=2):
        """使用IRIS检索最相关的文档"""
        # 获取查询的嵌入
        query_embedding = self._get_embedding([query])[0]
        query_embedding_str = ",".join(map(str, query_embedding))
        #print(query_embedding_str)
        # 使用IRIS的向量相似度搜索功能
        # 假设IRIS已启用向量搜索功能，这里使用余弦相似度
        cursor = self.connection.cursor()
        sqlStr = """
            SELECT TOP ? RuleInsurance, VECTOR_DOT_PRODUCT(TO_VECTOR(?,float),DrugEmbedding) AS Similarity
            FROM Demo.DrugInfo
            ORDER BY Similarity DESC
            """
        cursor.execute(
            sqlStr,
            [top_k,query_embedding_str]
        )
        results = cursor.fetchall()
        #print("查询结果:")
        #print(results)
        cursor.close()
        # 处理结果
        retrieved_docs = []
        for row in results:
            retrieved_docs.append({
                "RuleInsurance": row[0],
                "score": row[1]  # 相似度分数
            })
        print(retrieved_docs)
        return retrieved_docs


# 使用示例
if __name__ == "__main__":
    # 创建RAG系统实例（请根据您的IRIS配置修改参数）
    rag_system = IRISRAG(
        host="localhost",
        port=1980,
        namespace="MCP",
        username="superuser",
        password="SYS"
    )
    
    try:
        # 添加文档到向量库
        documents = [
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "大语言模型是一种人工智能模型，旨在理解和生成人类语言。它们在大量文本数据上进行训练，可以执行广泛的任务。",
            "RAG（Retrieval-Augmented Generation）是一种结合了信息检索技术与语言生成模型的人工智能技术。",
            "向量数据库是一种专门用于存储和检索高维向量的数据库系统，常用于相似性搜索和推荐系统。",
            "Qwen-Plus是通义千问系列中的高性能语言模型，适用于各种复杂、多步骤的任务。"
        ]
        #rag_system.add_documents(documents)
        
        # 执行RAG查询
        #query = "什么是RAG技术？"
        query = "左奥硝唑氯化钠"
        #result = rag_system.rag_query(query)
        result = rag_system.drug_query(query,5)
        #print(result)
        # 输出结果
        print(f"问题: {query}\n")
        print(f"答案: {result['answer']}\n")
        """
        print("参考文档:")
        for i, doc in enumerate(result["sources"]):
            print(f"文档 {i+1} (相似度: {doc['score']}):")
            print(f"{doc['RuleInsurance']}\n")
        """

    finally:
        # 确保关闭连接
        rag_system.close()