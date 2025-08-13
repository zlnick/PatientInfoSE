import os
from dotenv import load_dotenv
import numpy as np
from dashscope import TextEmbedding, Generation
# 导入InterSystems IRIS Python驱动
import iris as irisnative

load_dotenv()

class IRISRAG:
    def __init__(self, host="localhost", port=1980, namespace="MCP", username="superuser", password="SYS"):
        print(os.getenv("DASHSCOPE_API_KEY"))
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
        print(context_docs)
        context = "\n\n".join([doc["RuleInsurance"] for doc in context_docs])
        prompt = f"""基于以下上下文信息回答医保拒付风险相关的问题：

{context}

问题：{query}

患者信息:{patInfo}

要依据获取到的医保规则仔细分析，确认约束是不是都得到了满足。
例如盐酸右美托咪定的报销条件为：成人术前镇静/抗焦虑，则只有患者为成人且有手术医嘱且术前有类似焦虑的诊断或并病程记录才算满足条件。
如果上下文中没有明确的医保报销约束，则根据你的常识判断。
回答时要简洁，按顺序包括三个方面内容：
1. 结论：根据当前上下文中的信息明确回答是否有医保拒付风险
2. 规则：上下文中获得的医保规则是什么样的
3. 解释：如果有拒付风险，说明原因。如果没有拒付风险，说明支撑条件是什么。

回答："""
        #print(prompt)
        # 调用Qwen-Plus生成答案
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
        relevant_docs = self.drug_retrieve(query, top_k)
        
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
        #print(results)
        cursor.close()
        # 处理结果
        retrieved_docs = []
        for row in results:
            retrieved_docs.append({
                "RuleInsurance": row[0],
                "score": row[1]  # 相似度分数
            })
        
        return retrieved_docs

tested_patient = """
{
    "resourceType": "Bundle",
    "id": "047955c2-77fd-11f0-9612-0242ac140003",
    "type": "searchset",
    "timestamp": "2025-08-13T04:21:43Z",
    "link": [
        {
            "relation": "self",
            "url": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Patient/794/$everything"
        }
    ],
    "entry": [
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Patient/794",
            "resource": {
                "resourceType": "Patient",
                "id": "794",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"
                    ],
                    "lastUpdated": "2025-06-03T07:04:45Z",
                    "versionId": "1"
                },
                "text": {
                    "status": "generated",
                    "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Generated by <a href=\"https://github.com/synthetichealth/synthea\">Synthea</a>.Version identifier: master-branch-latest\n .   Person seed: 6428298403214954790  Population seed: 1748932747118</div>"
                },
                "extension": [
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                        "extension": [
                            {
                                "url": "ombCategory",
                                "valueCoding": {
                                    "system": "urn:oid:2.16.840.1.113883.6.238",
                                    "code": "2054-5",
                                    "display": "Black or African American"
                                }
                            },
                            {
                                "url": "text",
                                "valueString": "Black or African American"
                            }
                        ]
                    },
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                        "extension": [
                            {
                                "url": "ombCategory",
                                "valueCoding": {
                                    "system": "urn:oid:2.16.840.1.113883.6.238",
                                    "code": "2186-5",
                                    "display": "Not Hispanic or Latino"
                                }
                            },
                            {
                                "url": "text",
                                "valueString": "Not Hispanic or Latino"
                            }
                        ]
                    },
                    {
                        "url": "http://hl7.org/fhir/StructureDefinition/patient-mothersMaidenName",
                        "valueString": "欣然 丁"
                    },
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex",
                        "valueCode": "M"
                    },
                    {
                        "url": "http://hl7.org/fhir/StructureDefinition/patient-birthPlace",
                        "valueAddress": {
                            "city": "Maynard",
                            "state": "Massachusetts",
                            "country": "US"
                        }
                    },
                    {
                        "url": "http://synthetichealth.github.io/synthea/disability-adjusted-life-years",
                        "valueDecimal": 0.18519836746375412
                    },
                    {
                        "url": "http://synthetichealth.github.io/synthea/quality-adjusted-life-years",
                        "valueDecimal": 66.81480163253624
                    }
                ],
                "identifier": [
                    {
                        "system": "https://github.com/synthetichealth/synthea",
                        "value": "e7cb4686-bbab-7568-cb01-457c5e212c86"
                    },
                    {
                        "type": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                    "code": "MR",
                                    "display": "Medical Record Number"
                                }
                            ],
                            "text": "Medical Record Number"
                        },
                        "system": "http://hospital.smarthealthit.org",
                        "value": "e7cb4686-bbab-7568-cb01-457c5e212c86"
                    },
                    {
                        "type": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                    "code": "SS",
                                    "display": "Social Security Number"
                                }
                            ],
                            "text": "Social Security Number"
                        },
                        "system": "http://hl7.org/fhir/sid/us-ssn",
                        "value": "999-13-6876"
                    },
                    {
                        "type": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                    "code": "DL",
                                    "display": "Driver's license number"
                                }
                            ],
                            "text": "Driver's license number"
                        },
                        "system": "urn:oid:2.16.840.1.113883.4.3.25",
                        "value": "S99914803"
                    },
                    {
                        "type": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                    "code": "PPN",
                                    "display": "Passport Number"
                                }
                            ],
                            "text": "Passport Number"
                        },
                        "system": "http://hl7.org/fhir/sid/passport-USA",
                        "value": "X80823618X"
                    }
                ],
                "name": [
                    {
                        "use": "official",
                        "family": "韩",
                        "given": [
                            "伟泽"
                        ],
                        "prefix": [
                            "Mr."
                        ]
                    }
                ],
                "telecom": [
                    {
                        "system": "phone",
                        "value": "555-873-1430",
                        "use": "home"
                    }
                ],
                "gender": "male",
                "birthDate": "1957-07-06",
                "address": [
                    {
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/geolocation",
                                "extension": [
                                    {
                                        "url": "latitude",
                                        "valueDecimal": 42.107978311855355
                                    },
                                    {
                                        "url": "longitude",
                                        "valueDecimal": -71.93300216625047
                                    }
                                ]
                            }
                        ],
                        "line": [
                            "915 孙 Hollow"
                        ],
                        "city": "Charlton",
                        "state": "MA",
                        "postalCode": "00000",
                        "country": "US"
                    }
                ],
                "maritalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus",
                            "code": "D",
                            "display": "Divorced"
                        }
                    ],
                    "text": "Divorced"
                },
                "multipleBirthBoolean": false,
                "communication": [
                    {
                        "language": {
                            "coding": [
                                {
                                    "system": "urn:ietf:bcp:47",
                                    "code": "en-US",
                                    "display": "English (United States)"
                                }
                            ],
                            "text": "English (United States)"
                        }
                    }
                ]
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/795",
            "resource": {
                "resourceType": "Encounter",
                "id": "795",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter"
                    ],
                    "lastUpdated": "2025-06-03T07:04:48Z",
                    "versionId": "1"
                },
                "identifier": [
                    {
                        "use": "official",
                        "system": "https://github.com/synthetichealth/synthea",
                        "value": "6a95f6de-b045-ca37-fd41-ff951669942b"
                    }
                ],
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "410620009",
                                "display": "Well child visit (procedure)"
                            }
                        ],
                        "text": "Well child visit (procedure)"
                    }
                ],
                "subject": {
                    "reference": "Patient/794",
                    "display": "Mr. 伟泽 韩"
                },
                "participant": [
                    {
                        "type": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                                        "code": "PPRF",
                                        "display": "primary performer"
                                    }
                                ],
                                "text": "primary performer"
                            }
                        ],
                        "period": {
                            "start": "1974-08-24T05:21:50+08:00",
                            "end": "1974-08-24T05:36:50+08:00"
                        },
                        "individual": {
                            "reference": "Practitioner/788",
                            "display": "Dr. 忠林 郭"
                        }
                    }
                ],
                "period": {
                    "start": "1974-08-24T05:21:50+08:00",
                    "end": "1974-08-24T05:36:50+08:00"
                },
                "location": [
                    {
                        "location": {
                            "reference": "Location/773",
                            "display": "RENAISSANCE PRIMARY CARE LLC"
                        }
                    }
                ],
                "serviceProvider": {
                    "reference": "Organization/772",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Practitioner/788",
            "resource": {
                "resourceType": "Practitioner",
                "id": "788",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-practitioner"
                    ],
                    "lastUpdated": "2025-06-03T07:04:14Z",
                    "versionId": "1"
                },
                "extension": [
                    {
                        "url": "http://synthetichealth.github.io/synthea/utilization-encounters-extension",
                        "valueInteger": 57
                    }
                ],
                "identifier": [
                    {
                        "system": "http://hl7.org/fhir/sid/us-npi",
                        "value": "9999949495"
                    }
                ],
                "active": true,
                "name": [
                    {
                        "family": "郭",
                        "given": [
                            "忠林"
                        ],
                        "prefix": [
                            "Dr."
                        ]
                    }
                ],
                "telecom": [
                    {
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-direct",
                                "valueBoolean": true
                            }
                        ],
                        "system": "email",
                        "value": "忠林.郭@example.com",
                        "use": "work"
                    }
                ],
                "address": [
                    {
                        "line": [
                            "11-15 SANDERSDALE ROAD"
                        ],
                        "city": "WORCESTER",
                        "state": "MA",
                        "postalCode": "016032467",
                        "country": "US"
                    }
                ],
                "gender": "male"
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/796",
            "resource": {
                "resourceType": "Condition",
                "id": "796",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition-encounter-diagnosis"
                    ],
                    "lastUpdated": "2025-06-03T07:04:45Z",
                    "versionId": "1"
                },
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed"
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
                            "system": "http://snomed.info/sct",
                            "code": "160968000",
                            "display": "Risk activity involvement (finding)"
                        }
                    ],
                    "text": "Risk activity involvement (finding)"
                },
                "subject": {
                    "reference": "Patient/794"
                },
                "encounter": {
                    "reference": "Encounter/795"
                },
                "onsetDateTime": "1974-08-24T06:36:35+08:00",
                "recordedDate": "1974-08-24T06:36:35+08:00"
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Claim/799",
            "resource": {
                "resourceType": "Claim",
                "id": "799",
                "status": "active",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                            "code": "professional"
                        }
                    ]
                },
                "use": "claim",
                "patient": {
                    "reference": "Patient/794",
                    "display": "伟泽 韩"
                },
                "billablePeriod": {
                    "start": "1974-08-24T05:21:50+08:00",
                    "end": "1974-08-24T05:36:50+08:00"
                },
                "created": "1974-08-24T05:36:50+08:00",
                "provider": {
                    "reference": "Organization/772",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "priority": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/processpriority",
                            "code": "normal"
                        }
                    ]
                },
                "facility": {
                    "reference": "Location/773",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "diagnosis": [
                    {
                        "sequence": 1,
                        "diagnosisReference": {
                            "reference": "Condition/796"
                        }
                    }
                ],
                "insurance": [
                    {
                        "sequence": 1,
                        "focal": true,
                        "coverage": {
                            "display": "Cigna Health"
                        }
                    }
                ],
                "item": [
                    {
                        "sequence": 1,
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "410620009",
                                    "display": "Well child visit (procedure)"
                                }
                            ],
                            "text": "Well child visit (procedure)"
                        },
                        "encounter": [
                            {
                                "reference": "Encounter/795"
                            }
                        ]
                    },
                    {
                        "sequence": 2,
                        "diagnosisSequence": [
                            1
                        ],
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "160968000",
                                    "display": "Risk activity involvement (finding)"
                                }
                            ],
                            "text": "Risk activity involvement (finding)"
                        }
                    }
                ],
                "total": {
                    "value": 1451.41,
                    "currency": "USD"
                },
                "meta": {
                    "lastUpdated": "2025-06-03T07:04:50Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/ExplanationOfBenefit/800",
            "resource": {
                "resourceType": "ExplanationOfBenefit",
                "id": "800",
                "contained": [
                    {
                        "resourceType": "ServiceRequest",
                        "id": "referral",
                        "status": "completed",
                        "intent": "order",
                        "subject": {
                            "reference": "urn:uuid:e7cb4686-bbab-7568-cb01-457c5e212c86"
                        },
                        "requester": {
                            "reference": "Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999949495"
                        },
                        "performer": [
                            {
                                "reference": "Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999949495"
                            }
                        ]
                    },
                    {
                        "resourceType": "Coverage",
                        "id": "coverage",
                        "status": "active",
                        "type": {
                            "text": "Cigna Health"
                        },
                        "beneficiary": {
                            "reference": "urn:uuid:e7cb4686-bbab-7568-cb01-457c5e212c86"
                        },
                        "payor": [
                            {
                                "display": "Cigna Health"
                            }
                        ]
                    }
                ],
                "identifier": [
                    {
                        "system": "https://bluebutton.cms.gov/resources/variables/clm_id",
                        "value": "b3b90798-f0ad-5e4e-94bb-6da10cb9e527"
                    },
                    {
                        "system": "https://bluebutton.cms.gov/resources/identifier/claim-group",
                        "value": "99999999999"
                    }
                ],
                "status": "active",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                            "code": "professional"
                        }
                    ]
                },
                "use": "claim",
                "patient": {
                    "reference": "Patient/794"
                },
                "billablePeriod": {
                    "start": "1974-08-24T05:36:50+08:00",
                    "end": "1975-08-24T05:36:50+08:00"
                },
                "created": "1974-08-24T05:36:50+08:00",
                "insurer": {
                    "display": "Cigna Health"
                },
                "provider": {
                    "reference": "Practitioner/788"
                },
                "referral": {
                    "reference": "#referral"
                },
                "facility": {
                    "reference": "Location/773",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "claim": {
                    "reference": "Claim/799"
                },
                "outcome": "complete",
                "careTeam": [
                    {
                        "sequence": 1,
                        "provider": {
                            "reference": "Practitioner/788"
                        },
                        "role": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/claimcareteamrole",
                                    "code": "primary",
                                    "display": "Primary provider"
                                }
                            ]
                        }
                    }
                ],
                "diagnosis": [
                    {
                        "sequence": 1,
                        "diagnosisReference": {
                            "reference": "Condition/796"
                        },
                        "type": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/ex-diagnosistype",
                                        "code": "principal"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "insurance": [
                    {
                        "focal": true,
                        "coverage": {
                            "reference": "#coverage",
                            "display": "Cigna Health"
                        }
                    }
                ],
                "item": [
                    {
                        "sequence": 1,
                        "category": {
                            "coding": [
                                {
                                    "system": "https://bluebutton.cms.gov/resources/variables/line_cms_type_srvc_cd",
                                    "code": "1",
                                    "display": "Medical care"
                                }
                            ]
                        },
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "410620009",
                                    "display": "Well child visit (procedure)"
                                }
                            ],
                            "text": "Well child visit (procedure)"
                        },
                        "servicedPeriod": {
                            "start": "1974-08-24T05:21:50+08:00",
                            "end": "1974-08-24T05:36:50+08:00"
                        },
                        "locationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/ex-serviceplace",
                                    "code": "19",
                                    "display": "Off Campus-Outpatient Hospital"
                                }
                            ]
                        },
                        "encounter": [
                            {
                                "reference": "Encounter/795"
                            }
                        ]
                    },
                    {
                        "sequence": 2,
                        "diagnosisSequence": [
                            1
                        ],
                        "category": {
                            "coding": [
                                {
                                    "system": "https://bluebutton.cms.gov/resources/variables/line_cms_type_srvc_cd",
                                    "code": "1",
                                    "display": "Medical care"
                                }
                            ]
                        },
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "160968000",
                                    "display": "Risk activity involvement (finding)"
                                }
                            ],
                            "text": "Risk activity involvement (finding)"
                        },
                        "servicedPeriod": {
                            "start": "1974-08-24T05:21:50+08:00",
                            "end": "1974-08-24T05:36:50+08:00"
                        },
                        "locationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/ex-serviceplace",
                                    "code": "19",
                                    "display": "Off Campus-Outpatient Hospital"
                                }
                            ]
                        }
                    }
                ],
                "total": [
                    {
                        "category": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/adjudication",
                                    "code": "submitted",
                                    "display": "Submitted Amount"
                                }
                            ],
                            "text": "Submitted Amount"
                        },
                        "amount": {
                            "value": 1451.41,
                            "currency": "USD"
                        }
                    }
                ],
                "payment": {
                    "amount": {
                        "value": 0.0,
                        "currency": "USD"
                    }
                },
                "meta": {
                    "lastUpdated": "2025-06-03T07:04:50Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/801",
            "resource": {
                "resourceType": "Encounter",
                "id": "801",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter"
                    ],
                    "lastUpdated": "2025-06-03T07:04:51Z",
                    "versionId": "1"
                },
                "identifier": [
                    {
                        "use": "official",
                        "system": "https://github.com/synthetichealth/synthea",
                        "value": "9435cce4-5a6e-933a-4355-963e6870d2e1"
                    }
                ],
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "162673000",
                                "display": "General examination of patient (procedure)"
                            }
                        ],
                        "text": "General examination of patient (procedure)"
                    }
                ],
                "subject": {
                    "reference": "Patient/794",
                    "display": "Mr. 伟泽 韩"
                },
                "participant": [
                    {
                        "type": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                                        "code": "PPRF",
                                        "display": "primary performer"
                                    }
                                ],
                                "text": "primary performer"
                            }
                        ],
                        "period": {
                            "start": "1975-08-30T05:21:50+08:00",
                            "end": "1975-08-30T06:00:15+08:00"
                        },
                        "individual": {
                            "reference": "Practitioner/788",
                            "display": "Dr. 忠林 郭"
                        }
                    }
                ],
                "period": {
                    "start": "1975-08-30T05:21:50+08:00",
                    "end": "1975-08-30T06:00:15+08:00"
                },
                "location": [
                    {
                        "location": {
                            "reference": "Location/773",
                            "display": "RENAISSANCE PRIMARY CARE LLC"
                        }
                    }
                ],
                "serviceProvider": {
                    "reference": "Organization/772",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Condition/802",
            "resource": {
                "resourceType": "Condition",
                "id": "802",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition-encounter-diagnosis"
                    ],
                    "lastUpdated": "2025-06-03T07:04:45Z",
                    "versionId": "1"
                },
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active"
                        }
                    ]
                },
                "verificationStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                            "code": "confirmed"
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
                            "system": "http://snomed.info/sct",
                            "code": "224299000",
                            "display": "Received higher education (finding)"
                        }
                    ],
                    "text": "Received higher education (finding)"
                },
                "subject": {
                    "reference": "Patient/794"
                },
                "encounter": {
                    "reference": "Encounter/801"
                },
                "onsetDateTime": "1975-08-30T06:00:15+08:00",
                "recordedDate": "1975-08-30T06:00:15+08:00"
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Claim/805",
            "resource": {
                "resourceType": "Claim",
                "id": "805",
                "status": "active",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                            "code": "professional"
                        }
                    ]
                },
                "use": "claim",
                "patient": {
                    "reference": "Patient/794",
                    "display": "伟泽 韩"
                },
                "billablePeriod": {
                    "start": "1975-08-30T05:21:50+08:00",
                    "end": "1975-08-30T06:00:15+08:00"
                },
                "created": "1975-08-30T06:00:15+08:00",
                "provider": {
                    "reference": "Organization/772",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "priority": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/processpriority",
                            "code": "normal"
                        }
                    ]
                },
                "facility": {
                    "reference": "Location/773",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "diagnosis": [
                    {
                        "sequence": 1,
                        "diagnosisReference": {
                            "reference": "Condition/802"
                        }
                    }
                ],
                "insurance": [
                    {
                        "sequence": 1,
                        "focal": true,
                        "coverage": {
                            "display": "Cigna Health"
                        }
                    }
                ],
                "item": [
                    {
                        "sequence": 1,
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "162673000",
                                    "display": "General examination of patient (procedure)"
                                }
                            ],
                            "text": "General examination of patient (procedure)"
                        },
                        "encounter": [
                            {
                                "reference": "Encounter/801"
                            }
                        ]
                    },
                    {
                        "sequence": 2,
                        "diagnosisSequence": [
                            1
                        ],
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "224299000",
                                    "display": "Received higher education (finding)"
                                }
                            ],
                            "text": "Received higher education (finding)"
                        }
                    }
                ],
                "total": {
                    "value": 1081.28,
                    "currency": "USD"
                },
                "meta": {
                    "lastUpdated": "2025-06-03T07:04:51Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/ExplanationOfBenefit/806",
            "resource": {
                "resourceType": "ExplanationOfBenefit",
                "id": "806",
                "contained": [
                    {
                        "resourceType": "ServiceRequest",
                        "id": "referral",
                        "status": "completed",
                        "intent": "order",
                        "subject": {
                            "reference": "urn:uuid:e7cb4686-bbab-7568-cb01-457c5e212c86"
                        },
                        "requester": {
                            "reference": "Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999949495"
                        },
                        "performer": [
                            {
                                "reference": "Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999949495"
                            }
                        ]
                    },
                    {
                        "resourceType": "Coverage",
                        "id": "coverage",
                        "status": "active",
                        "type": {
                            "text": "Cigna Health"
                        },
                        "beneficiary": {
                            "reference": "urn:uuid:e7cb4686-bbab-7568-cb01-457c5e212c86"
                        },
                        "payor": [
                            {
                                "display": "Cigna Health"
                            }
                        ]
                    }
                ],
                "identifier": [
                    {
                        "system": "https://bluebutton.cms.gov/resources/variables/clm_id",
                        "value": "99cc81eb-ceee-90a4-9acc-d9871e9e7423"
                    },
                    {
                        "system": "https://bluebutton.cms.gov/resources/identifier/claim-group",
                        "value": "99999999999"
                    }
                ],
                "status": "active",
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                            "code": "professional"
                        }
                    ]
                },
                "use": "claim",
                "patient": {
                    "reference": "Patient/794"
                },
                "billablePeriod": {
                    "start": "1975-08-30T06:00:15+08:00",
                    "end": "1976-08-30T06:00:15+08:00"
                },
                "created": "1975-08-30T06:00:15+08:00",
                "insurer": {
                    "display": "Cigna Health"
                },
                "provider": {
                    "reference": "Practitioner/788"
                },
                "referral": {
                    "reference": "#referral"
                },
                "facility": {
                    "reference": "Location/773",
                    "display": "RENAISSANCE PRIMARY CARE LLC"
                },
                "claim": {
                    "reference": "Claim/805"
                },
                "outcome": "complete",
                "careTeam": [
                    {
                        "sequence": 1,
                        "provider": {
                            "reference": "Practitioner/788"
                        },
                        "role": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/claimcareteamrole",
                                    "code": "primary",
                                    "display": "Primary provider"
                                }
                            ]
                        }
                    }
                ],
                "diagnosis": [
                    {
                        "sequence": 1,
                        "diagnosisReference": {
                            "reference": "Condition/802"
                        },
                        "type": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/ex-diagnosistype",
                                        "code": "principal"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "insurance": [
                    {
                        "focal": true,
                        "coverage": {
                            "reference": "#coverage",
                            "display": "Cigna Health"
                        }
                    }
                ],
                "item": [
                    {
                        "sequence": 1,
                        "category": {
                            "coding": [
                                {
                                    "system": "https://bluebutton.cms.gov/resources/variables/line_cms_type_srvc_cd",
                                    "code": "1",
                                    "display": "Medical care"
                                }
                            ]
                        },
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "162673000",
                                    "display": "General examination of patient (procedure)"
                                }
                            ],
                            "text": "General examination of patient (procedure)"
                        },
                        "servicedPeriod": {
                            "start": "1975-08-30T05:21:50+08:00",
                            "end": "1975-08-30T06:00:15+08:00"
                        },
                        "locationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/ex-serviceplace",
                                    "code": "19",
                                    "display": "Off Campus-Outpatient Hospital"
                                }
                            ]
                        },
                        "encounter": [
                            {
                                "reference": "Encounter/801"
                            }
                        ]
                    },
                    {
                        "sequence": 2,
                        "diagnosisSequence": [
                            1
                        ],
                        "category": {
                            "coding": [
                                {
                                    "system": "https://bluebutton.cms.gov/resources/variables/line_cms_type_srvc_cd",
                                    "code": "1",
                                    "display": "Medical care"
                                }
                            ]
                        },
                        "productOrService": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "224299000",
                                    "display": "Received higher education (finding)"
                                }
                            ],
                            "text": "Received higher education (finding)"
                        },
                        "servicedPeriod": {
                            "start": "1975-08-30T05:21:50+08:00",
                            "end": "1975-08-30T06:00:15+08:00"
                        },
                        "locationCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/ex-serviceplace",
                                    "code": "19",
                                    "display": "Off Campus-Outpatient Hospital"
                                }
                            ]
                        }
                    }
                ],
                "total": [
                    {
                        "category": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/adjudication",
                                    "code": "submitted",
                                    "display": "Submitted Amount"
                                }
                            ],
                            "text": "Submitted Amount"
                        },
                        "amount": {
                            "value": 1081.28,
                            "currency": "USD"
                        }
                    }
                ],
                "payment": {
                    "amount": {
                        "value": 0.0,
                        "currency": "USD"
                    }
                },
                "meta": {
                    "lastUpdated": "2025-06-03T07:04:51Z",
                    "versionId": "1"
                }
            },
            "search": {
                "mode": "include"
            }
        },
        {
            "fullUrl": "http://localhost:52880/csp/healthshare/fhirserver/fhir/r4/Encounter/807",
            "resource": {
                "resourceType": "Encounter",
                "id": "807",
                "meta": {
                    "profile": [
                        "http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter"
                    ],
                    "lastUpdated": "2025-06-03T07:04:51Z",
                    "versionId": "1"
                },
                "identifier": [
                    {
                        "use": "official",
                        "system": "https://github.com/synthetichealth/synthea",
                        "value": "74fc0953-48b8-d8c2-5356-f80b27a4c160"
                    }
                ],
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "IMP"
                },
                "type": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "32485007",
                                "display": "住院就诊"
                            }
                        ],
                        "text": "住院就诊"
                    }
                ],
                "subject": {
                    "reference": "Patient/794",
                    "display": "Mr. 伟泽 韩"
                },
                "participant": [
                    {
                        "type": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                                        "code": "PPRF",
                                        "display": "primary performer"
                                    }
                                ],
                                "text": "primary performer"
                            }
                        ],
                        "period": {
                            "start": "1987-07-11T06:21:50+09:00",
                            "end": "1987-07-12T06:21:50+09:00"
                        },
                        "individual": {
                            "reference": "Practitioner/786",
                            "display": "Dr. 嘉轩 钱"
                        }
                    }
                ],
                "period": {
                    "start": "1987-07-11T06:21:50+09:00",
                    "end": "1987-07-12T06:21:50+09:00"
                },
                "location": [
                    {
                        "location": {
                            "reference": "Location/771",
                            "display": "VIBRA HOSPITAL OF WESTERN MASSACHUSETTS LLC"
                        }
                    }
                ],
                "serviceProvider": {
                    "reference": "Organization/770",
                    "display": "VIBRA HOSPITAL OF WESTERN MASSACHUSETTS LLC"
                }
            },
            "search": {
                "mode": "include"
            }
        }
    ]
}
"""


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
        query = "水合氯醛灌肠剂"
        #result = rag_system.rag_query(query)
        result = rag_system.drug_query(query,5)
        print(result)
        # 输出结果
        print(f"问题: {query}\n")
        print(f"答案: {result['answer']}\n")
        print("参考文档:")
        for i, doc in enumerate(result["sources"]):
            print(f"文档 {i+1} (相似度: {doc['score']}):")
            print(f"{doc['RuleInsurance']}\n")
    
    finally:
        # 确保关闭连接
        rag_system.close()