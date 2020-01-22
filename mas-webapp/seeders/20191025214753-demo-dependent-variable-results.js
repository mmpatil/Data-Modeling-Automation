'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.bulkInsert('DependentVariableResult', [{
      RunId: 1,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    },
    {
      RunId: 1,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    },
    {
      RunId: 1,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    },
    {
      RunId: 4,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    },
    {
      RunId: 4,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    },
    {
      RunId: 4,
      Name: "DepV1",
      Coefficient: 5.2,
      Pval: 0.23,
      Transformations: "????",
      UnitRoot: "???"
    }], {});
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.bulkDelete('DependentVariableResult', null, {});
  }
};
